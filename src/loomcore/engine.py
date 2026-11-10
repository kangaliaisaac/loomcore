"""One run, while the driver holds the session lock: emit events, perform effects.

`Run` is the shell around `step`. It appends each event it produces, folds it into
the state, and performs the effects `step` returns until the run finishes or
pauses. The decisions are `step`'s and `check_limits`'s; nothing here chooses.
"""

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from pydantic import JsonValue

from loomcore.conversation import build_messages, call_hash, final_text, tool_call
from loomcore.eventlog import EventWriter
from loomcore.events import Event, JsonObject, ModelCall, RunStarted, make_event
from loomcore.limits import LimitReached, Limits, check_limits, elapsed
from loomcore.ports import (
    Config,
    Decision,
    ModelCallError,
    ModelRequest,
    Ports,
    RunRef,
    RunResult,
)
from loomcore.replay import RunState
from loomcore.retry import GiveUp, RetryPolicy, decide
from loomcore.spend import append_spend, month_spend, read_spend, run_key, spend_line
from loomcore.step import AskHuman, CallModel, Effect, Fail, RunTool, Stop, step

_LOG = logging.getLogger(__name__)


class ResumeRefusedError(Exception):
    """`resume` was called in a state it cannot continue from; the message says how."""


class LoopStalledError(Exception):
    """Every effect was performed, yet the run is neither finished nor paused."""


@dataclass
class Run:
    config: Config
    ports: Ports
    retry: RetryPolicy
    spend_path: Path
    writer: EventWriter
    ref: RunRef
    limits: Limits
    chain: list[Event]  # the events replay reads for this run
    state: RunState = field(default_factory=RunState)
    next_seq: int = 0

    def emit(self, kind: str, **fields: object) -> Event:
        event = make_event(
            kind,
            seq=self.next_seq,
            timestamp=self.ports.clock(),
            session_id=self.ref.session_id,
            run_id=self.ref.run_id,
            **fields,
        )
        self.writer.append(event)
        self.chain.append(event)
        self.next_seq += 1
        return event

    def fold(self, events: list[Event]) -> list[Effect]:
        """Fold events in order; the effects are those of the last."""
        effects: list[Effect] = []
        for event in events:
            self.state, effects = step(self.state, event, self.ports.read_only)
        return effects

    def drive(self, effects: list[Effect]) -> RunResult:
        queue = list(effects)
        while queue:
            outcome = self._perform(queue.pop(0))
            if isinstance(outcome, RunResult):
                return outcome
            queue = self.fold(outcome) + queue
        raise LoopStalledError("no effect is left and the run is not stopped")

    def reopen(self, decision: Decision | None, limits: Limits | None) -> list[Effect]:
        """Append the `run_resumed` (and the decision) this state calls for."""
        if self.state.finished is not None:
            raise ResumeRefusedError(
                f"the run already finished ({self.state.finished})"
            )
        if self.state.paused:
            return self._resume_pause(decision, limits)
        if decision is not None or limits is not None:
            raise ResumeRefusedError("only a paused run takes a decision or limits")
        return self.fold([self._resumed("crash")])

    def _resume_pause(
        self, decision: Decision | None, limits: Limits | None
    ) -> list[Effect]:
        if self.state.paused_limit is not None:
            if limits is None or decision is not None:
                raise ResumeRefusedError(
                    f"paused on {self.state.paused_limit}: resume with limits only"
                )
            self.limits = limits
            return self.fold([self._resumed("limit", limits=limits.to_json())])
        if decision is None or limits is not None:
            raise ResumeRefusedError("paused for approval: resume with a decision only")
        asked = self.state.open_approvals[0]
        resumed = self._resumed("approval")
        kind = "approval_granted" if decision.granted else "approval_denied"
        answer = self.emit(kind, tool_use_id=asked, by=decision.by, note=decision.note)
        return self.fold([resumed, answer])

    def _resumed(self, reason: str, **fields: object) -> Event:
        return self.emit(
            "run_resumed", reason=reason, resume_from=self.next_seq - 1, **fields
        )

    def _perform(self, effect: Effect) -> list[Event] | RunResult:
        outcome: list[Event] | RunResult
        match effect:
            case CallModel():
                outcome = self._call_model()
            case RunTool(tool_use_id):
                outcome = self._run_tool(tool_use_id)
            case Fail(tool_use_id, reason):
                outcome = [self._complete(tool_use_id, ok=False, content=reason)]
            case AskHuman(tool_use_id, question):
                outcome = self._ask(tool_use_id, question)
            case Stop():
                outcome = self._finish(effect)
        return outcome

    def _call_model(self) -> list[Event] | RunResult:
        request = self._request()
        reached = check_limits(
            self.state,
            self.limits,
            self._month_to_date(),
            self.ports.pricing.worst_case(request),
            self.ports.clock(),
        )
        if isinstance(reached, LimitReached):
            return self._pause("limit", reached.limit)
        started = self.ports.monotonic()
        message = self._with_retries(request)
        if isinstance(message, RunResult):
            return message
        # After the response, so a pause or a give-up during retries leaves no
        # turn behind that max_turns would count.
        turn = self.emit("turn_started", turn=self.state.turns + 1)
        cost, version = self.ports.pricing.cost(message)
        call = self.emit(
            "model_call",
            type="assistant",
            message=message,
            duration_ms=self._since(started),
            cost_usd=cost,
            pricing_version=version,
        )
        assert isinstance(call, ModelCall)
        append_spend(self.spend_path, spend_line(call))
        return [turn, call]

    def _with_retries(self, request: ModelRequest) -> JsonObject | RunResult:
        attempt = 0
        while True:
            try:
                return self.ports.model(request)
            except ModelCallError as failure:
                verdict = decide(attempt, failure.error, self.ports.random, self.retry)
            if isinstance(verdict, GiveUp):
                return self._finish(Stop("error", verdict.reason))
            # retry-after is uncapped, so sleeping may outlast the run's timeout
            wake = timedelta(seconds=verdict.seconds)
            if elapsed(self.state, self.ports.clock()) + wake >= timedelta(
                seconds=self.limits.timeout
            ):
                return self._pause("limit", "timeout")
            self.ports.sleep(verdict.seconds)
            attempt += 1

    def _request(self) -> ModelRequest:
        root = next(e for e in self.chain if isinstance(e, RunStarted))
        return ModelRequest(
            model=self.config.model,
            system=self.config.system,
            messages=build_messages(root.task, self.chain),
            tools=self.config.tools,
            max_tokens=self.limits.max_output_tokens,
        )

    def _month_to_date(self) -> Decimal:
        key = run_key(self.ref.session_id, self.ref.run_id)
        spend = month_spend(read_spend(self.spend_path), key, self.ports.clock())
        for line in spend.skipped:
            _LOG.warning("spend log line not counted: %r", line)
        return spend.usd

    def _run_tool(self, tool_use_id: str) -> list[Event]:
        name, args = tool_call(self.chain, tool_use_id)
        events: list[Event] = []
        if tool_use_id not in dict(self.state.requested):
            events.append(
                self.emit(
                    "tool_requested",
                    tool_use_id=tool_use_id,
                    name=name,
                    input=args,
                    call_hash=call_hash(name, args),
                )
            )
        started = self.ports.monotonic()
        result = self.ports.tool(name, args)
        done = self._complete(
            tool_use_id, ok=result.ok, content=result.content, ms=self._since(started)
        )
        return [*events, done]

    def _complete(
        self, tool_use_id: str, *, ok: bool, content: JsonValue, ms: int = 0
    ) -> Event:
        return self.emit(
            "tool_completed",
            tool_use_id=tool_use_id,
            ok=ok,
            content=content,
            duration_ms=ms,
            cached=False,
        )

    def _ask(self, tool_use_id: str, question: str) -> RunResult:
        if tool_use_id not in self.state.open_approvals:
            requested = self.emit(
                "approval_requested", tool_use_id=tool_use_id, question=question
            )
            self.fold([requested])
        return self._pause("awaiting_approval", None, question)

    def _pause(
        self, reason: str, limit: str | None, question: str | None = None
    ) -> RunResult:
        paused = self.emit(
            "run_paused", reason=reason, limit=limit, resume_from=self.next_seq - 1
        )
        self.fold([paused])
        return RunResult(self.ref, "paused", reason, limit, question)

    def _finish(self, stop: Stop) -> RunResult:
        """End the run: settle every pending call first, then write run_finished."""
        for tool_use_id in self.state.pending_tools:
            self._complete(tool_use_id, ok=False, content=f"run ended: {stop.reason}")
        result = stop.result if stop.result is not None else self._last_text()
        finished = self.emit("run_finished", reason=stop.reason, result=result)
        self.fold([finished])
        return RunResult(self.ref, "finished", stop.reason, result=result)

    def _last_text(self) -> str:
        calls = [e for e in self.chain if isinstance(e, ModelCall)]
        return final_text(calls[-1].message) if calls else ""

    def _since(self, started: float) -> int:
        return int((self.ports.monotonic() - started) * 1000)
