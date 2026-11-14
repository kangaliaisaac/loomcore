"""One run, while the driver holds the session lock: emit events, perform effects.

`Run` is the shell around `step`. It appends each event it produces, folds it into
the state, and performs the effects `step` returns until the run finishes or
pauses. The decisions are `step`'s and `check_limits`'s; nothing here chooses.
"""

import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from pydantic import JsonValue

from loomcore import ladder, schedule
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
from loomcore.notices import Notice
from loomcore.ports import (
    Config,
    Decision,
    ModelCallError,
    ModelRequest,
    Ports,
    RunRef,
    RunResult,
    ToolResult,
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

MAX_TOGETHER = 8  # calls of one wave running at once; the rest of the wave queues


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
            effect = queue.pop(0)
            if isinstance(effect, RunTool):
                # The turn's calls come together; their waves decide what overlaps.
                batch = [effect]
                while queue and isinstance(following := queue[0], RunTool):
                    batch.append(following)
                    queue.pop(0)
                outcome: list[Event] | RunResult = self._run_tools(batch)
            else:
                outcome = self._perform(effect)
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
            case RunTool():
                outcome = self._run_tools([effect])
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
        wave_of = self._waves(turn, unrequested)
        for k, (tool_use_id, name, digest, args) in enumerate(turn):
            if tool_use_id in unrequested:
                events.append(
                    self.emit(
                        "tool_requested",
                        tool_use_id=tool_use_id,
                        name=name,
                        input=args,
                        call_hash=digest,
                        wave=wave_of[k],
                    )
                )
        return events

    def _waves(
        self, turn: list[tuple[str, str, str, JsonObject]], unrequested: tuple[str, ...]
    ) -> dict[int, int]:
        """Which wave each call still to be requested runs in, from what its tool
        declared. Written into the log, so a resume never works it out again.

        Lines a crash left unwritten go after every wave already in the log, so a
        registry that changed since cannot put them beside a call they conflict with.
        """
        missing = [k for k, call in enumerate(turn) if call[0] in unrequested]
        written = dict(self.state.call_waves).values()
        base = max(written, default=-1) + 1
        grouped = schedule.waves([self.ports.access(turn[k][1]) for k in missing])
        return {
            missing[j]: base + wave for wave, calls in enumerate(grouped) for j in calls
        }

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

    def _run_tools(self, batch: list[RunTool]) -> list[Event]:
        """Run the calls, one wave after the other. The waves were written into the
        log with the `tool_requested` lines, so a resume reads the same ones."""
        waves = dict(self.state.call_waves)
        events: list[Event] = []
        for wave in sorted({waves[run.tool_use_id] for run in batch}):
            together = [run for run in batch if waves[run.tool_use_id] == wave]
            events += self._run_wave(together)
        return events

    def _run_wave(self, wave: list[RunTool]) -> list[Event]:
        if len(wave) == 1:
            return self._run_tool(wave[0])
        return self._run_together(wave)

    def _run_tool(self, run: RunTool) -> list[Event]:
        name, args = tool_call(self.chain, run.tool_use_id)
        started = self.ports.monotonic()
        try:
            with self.ports.notices.collecting() as bucket:
                result = self.ports.tool(name, args)
                ms = self._since(started)  # before the log writes below
        finally:
            # Whether the tool returned or broke, what it caused is logged, and a
            # refusal precedes the call's completion. The bucket was closed when the
            # block exited, so a notice that arrives later is not logged, only
            # warned about (spec 3.2).
            caused = self._emit_notices(run.tool_use_id, bucket.snapshot())
        return [*caused, self._completed(run, result, ms)]

    def _run_together(self, wave: list[RunTool]) -> list[Event]:
        """Run independent calls on threads. A thread only runs its tool and keeps
        what the call caused; the log has one writer, so this thread writes every
        event afterwards, each call's notices and then its completion, in the order
        the model emitted the calls."""
        caused: dict[str, tuple[Notice, ...]] = {}

        def work(tool_use_id: str) -> tuple[ToolResult, int]:
            name, args = tool_call(self.chain, tool_use_id)
            started = self.ports.monotonic()
            try:
                with self.ports.notices.collecting() as bucket:
                    result = self.ports.tool(name, args)
                    ms = self._since(started)
            finally:
                caused[tool_use_id] = bucket.snapshot()
            return result, ms

        with ThreadPoolExecutor(max_workers=min(len(wave), MAX_TOGETHER)) as pool:
            futures = [pool.submit(work, run.tool_use_id) for run in wave]
        events: list[Event] = []
        bug: BaseException | None = None
        for run, future in zip(wave, futures, strict=True):
            events += self._emit_notices(run.tool_use_id, caused[run.tool_use_id])
            error = future.exception()
            if error is None:
                result, ms = future.result()
                events.append(self._completed(run, result, ms))
            elif bug is None:
                bug = error  # the first, in the order the model emitted the calls
            else:
                _LOG.warning(
                    "%s also failed, after another call: %r", run.tool_use_id, error
                )
        if bug is not None:
            raise bug  # what the other calls caused and finished is already logged
        return events

    def _completed(self, run: RunTool, result: ToolResult, ms: int) -> Event:
        return self._complete(
            run.tool_use_id,
            ok=result.ok,
            content=_annotated(result.content, run.note),
            ms=ms,
            cached=result.cached,
        )

    def _emit_notices(self, tool_use_id: str, notices: Sequence[Notice]) -> list[Event]:
        return [
            self.emit(notice.kind, tool_use_id=tool_use_id, **notice.fields)
            for notice in notices
        ]

    def _complete(
        self,
        tool_use_id: str,
        *,
        ok: bool,
        content: JsonValue,
        ms: int = 0,
        cached: bool = False,
    ) -> Event:
        return self.emit(
            "tool_completed",
            tool_use_id=tool_use_id,
            ok=ok,
            content=content,
            duration_ms=ms,
            cached=cached,
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
