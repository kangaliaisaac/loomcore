"""One run, while the driver holds the session lock: emit events, perform effects.

`Run` is the shell around `step`. It appends each event it produces, folds it into
the state, and performs the effects `step` returns until the run finishes or
pauses. The decisions are `step`'s and `check_limits`'s; nothing here chooses.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from pydantic import JsonValue

from loomcore import ladder
from loomcore.conversation import build_messages, call_hash, final_text, tool_call
from loomcore.eventlog import EventWriter
from loomcore.events import (
    ApprovalRequested,
    Event,
    JsonObject,
    ModelCall,
    RunStarted,
    make_event,
)
from loomcore.limits import LimitReached, Limits, check_limits, elapsed
from loomcore.ports import (
    Config,
    Decision,
    ModelCallError,
    ModelRequest,
    Notice,
    Ports,
    RunRef,
    RunResult,
    ToolCrashedError,
)
from loomcore.replay import RunState, tool_use_ids
from loomcore.retry import GiveUp, RetryPolicy, decide
from loomcore.spend import append_spend, month_spend, read_spend, run_key, spend_line
from loomcore.step import (
    AskHuman,
    CallModel,
    Effect,
    Fail,
    RequestCalls,
    RunTool,
    Stop,
    step,
)

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
    switch_to: Config | None = None  # the configuration the switch rung changes to
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
            case RequestCalls(tool_use_ids):
                outcome = self._request_calls(tool_use_ids)
            case RunTool(tool_use_id, note):
                outcome = self._run_tool(tool_use_id, note)
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
        cost, version = self.ports.pricing.cost(
            message
        )  # before any event: it can fail
        turn = self.emit("turn_started", turn=self.state.turns + 1)
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

    def _request_calls(self, unrequested: tuple[str, ...]) -> list[Event]:
        """Judge the whole turn once, then write its missing `tool_requested` lines.

        The escalation, if any, goes first: a crash anywhere in the batch then
        leaves the rung in the log, and "every line written, no escalation" can only
        mean the turn was judged clean. A turn already judged is not judged again.
        """
        turn = self._turn_calls()
        events: list[Event] = []
        if self.state.escalated is None:
            hashes = self.state.call_hashes
            written = len(turn) - len(unrequested)  # their hashes are the last ones
            hits = ladder.tripped(
                hashes[: len(hashes) - written],
                [digest for _, _, digest, _ in turn],
                self.limits.repeat_window,
                self.limits.repeat_threshold,
            )
            if hits:
                events.append(self._escalate([turn[k][:3] for k in hits]))
        for tool_use_id, name, digest, args in turn:
            if tool_use_id in unrequested:
                events.append(
                    self.emit(
                        "tool_requested",
                        tool_use_id=tool_use_id,
                        name=name,
                        input=args,
                        call_hash=digest,
                    )
                )
        return events

    def _turn_calls(self) -> list[tuple[str, str, str, JsonObject]]:
        """The latest model_call's tool calls as (id, name, call_hash, input)."""
        latest = next(e for e in reversed(self.chain) if isinstance(e, ModelCall))
        calls = []
        for tool_use_id in tool_use_ids(latest.message):
            name, args = tool_call(self.chain, tool_use_id)
            calls.append((tool_use_id, name, call_hash(name, args), args))
        return calls

    def _escalate(self, hits: list[tuple[str, str, str]]) -> Event:
        """Climb one rung for the turn, however many calls tripped."""
        rung = min(self.state.rung + 1, ladder.STUCK_RUNG)
        first_id, name, digest = hits[0]
        fields: dict[str, object] = {}
        if rung == ladder.SWITCH_RUNG and self.switch_to is not None:
            self.config = self.switch_to
            fields = {
                "config": self.config.to_json(),
                "config_digest": self.config.digest(),
            }
        return self.emit(
            "escalation",
            rung=rung,
            reason=ladder.reason(
                name, self.limits.repeat_threshold, self.limits.repeat_window
            ),
            tool_use_id=first_id,
            call_hash=digest,
            tripped=[tool_use_id for tool_use_id, _, _ in hits],
            **fields,
        )

    def _run_tool(self, tool_use_id: str, note: str | None) -> list[Event]:
        name, args = tool_call(self.chain, tool_use_id)
        started = self.ports.monotonic()
        try:
            result = self.ports.tool(name, args)
        except ToolCrashedError as crash:
            self._emit_notices(tool_use_id, crash.notices)
            raise crash.error from crash
        ms = self._since(started)  # before the log writes below
        # What the call caused is written first, so a refusal precedes its completion.
        caused = self._emit_notices(tool_use_id, result.notices)
        content = _annotated(result.content, note)
        done = self._complete(tool_use_id, ok=result.ok, content=content, ms=ms)
        return [*caused, done]

    def _emit_notices(
        self, tool_use_id: str, notices: tuple[Notice, ...]
    ) -> list[Event]:
        return [
            self.emit(notice.kind, tool_use_id=tool_use_id, **notice.fields)
            for notice in notices
        ]

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
        if tool_use_id in self.state.open_approvals:
            question = self._logged_question(tool_use_id)
        else:
            requested = self.emit(
                "approval_requested", tool_use_id=tool_use_id, question=question
            )
            self.fold([requested])
        return self._pause("awaiting_approval", None, question)

    def _logged_question(self, tool_use_id: str) -> str:
        """The question the log already holds for an open approval."""
        return next(
            e.question
            for e in reversed(self.chain)
            if isinstance(e, ApprovalRequested) and e.tool_use_id == tool_use_id
        )

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
        settled = [self._settle_call(i, stop) for i in self.state.pending_tools]
        # A finish call's result stands even when it is null; other stops report the
        # model's last words.
        keep = stop.finish_id is not None or stop.result is not None
        result = stop.result if keep else self._last_text()
        finished = self.emit("run_finished", reason=stop.reason, result=result)
        self.fold([*settled, finished])
        return RunResult(self.ref, "finished", stop.reason, result=result)

    def _settle_call(self, tool_use_id: str, stop: Stop) -> Event:
        ok = tool_use_id == stop.finish_id  # the finish call itself succeeded
        content = "finished" if ok else f"run ended: {stop.reason}"
        return self._complete(tool_use_id, ok=ok, content=content)

    def _last_text(self) -> str:
        calls = [e for e in self.chain if isinstance(e, ModelCall)]
        return final_text(calls[-1].message) if calls else ""

    def _since(self, started: float) -> int:
        return int((self.ports.monotonic() - started) * 1000)


def _annotated(content: JsonValue, note: str | None) -> JsonValue:
    """A ladder message is added to the content the model will see."""
    if note is None:
        return content
    text = content if isinstance(content, str) else json.dumps(content)
    return f"{text}\n\n[{note}]"
