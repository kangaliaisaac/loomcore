"""The driver: run, resume and fork a run (spec section 1).

It opens the session under its lock, checks that a resume is allowed and that the
configuration matches, repairs the spend log, and hands the run to `engine.Run`.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from loomcore.engine import ResumeRefusedError, Run
from loomcore.eventlog import EventWriter, open_writer, read_events
from loomcore.events import Event, ForkOrigin, RunStarted
from loomcore.limits import Limits, limits_in_force
from loomcore.ports import Config, Decision, Ports, RunRef, RunResult, Task
from loomcore.replay import replay, run_events
from loomcore.retry import RetryPolicy
from loomcore.rewind import ensure_fork_point
from loomcore.spend import repair_spend, run_key

_LOG = logging.getLogger(__name__)


class ConfigChangedError(Exception):
    """The configuration this process holds is not the one the run started with."""


class SpendLogMissingError(Exception):
    """The spend log's directory does not exist; a mistyped path would read as $0."""


class Driver:
    """Runs a task, resumes a paused or crashed run, forks one from a clean point."""

    def __init__(
        self,
        config: Config,
        ports: Ports,
        session_path: Path,
        spend_path: Path,
        retry: RetryPolicy = RetryPolicy(),  # noqa: B008  (frozen, so shared safely)
    ) -> None:
        if not spend_path.parent.is_dir():
            raise SpendLogMissingError(
                f"{spend_path.parent} does not exist; create it or fix the path"
            )
        _LOG.info("spend log: %s", spend_path)
        self._config = config
        self._ports = ports
        self._session_path = session_path
        self._spend_path = spend_path
        self._retry = retry

    def run(self, task: Task) -> RunResult:
        with open_writer(self._session_path) as writer:
            run = self._new_run(writer, self._ports.new_run_id(), task.limits, [], [])
            started = run.emit(
                "run_started",
                task=task.prompt,
                config=self._config.to_json(),
                limits=task.limits.to_json(),
                config_digest=self._config.digest(),
            )
            return run.drive(run.fold([started]))

    def resume(
        self,
        ref: RunRef,
        decision: Decision | None = None,
        limits: Limits | None = None,
    ) -> RunResult:
        with open_writer(self._session_path) as writer:
            run = self._reopen(writer, ref)
            return run.drive(run.reopen(decision, limits))

    def fork(
        self, origin: ForkOrigin, hint: str | None = None, limits: Limits | None = None
    ) -> RunResult:
        with open_writer(self._session_path) as writer:
            events = list(read_events(self._session_path).events)
            ensure_fork_point(events, origin)
            chain = run_events(events, origin.run_id, upto=origin.seq)
            self._check_digest(chain)
            run = self._new_run(
                writer, self._ports.new_run_id(), limits_in_force(chain), events, chain
            )
            run.state = replay(chain)
            override = limits.to_json() if limits else None
            forked = run.emit("run_forked", from_=origin, hint=hint, limits=override)
            if limits:
                run.limits = limits
            return run.drive(run.fold([forked]))

    def _new_run(
        self,
        writer: EventWriter,
        run_id: str,
        limits: Limits,
        events: list[Event],
        chain: list[Event],
    ) -> Run:
        ref = RunRef(self._session_path.stem, run_id)
        return Run(
            config=self._config,
            ports=self._ports,
            retry=self._retry,
            spend_path=self._spend_path,
            writer=writer,
            ref=ref,
            limits=limits,
            events=events,
            chain=chain,
        )

    def _reopen(self, writer: EventWriter, ref: RunRef) -> Run:
        if ref.session_id != self._session_path.stem:
            raise ResumeRefusedError(
                f"{ref.session_id} is not the session in {self._session_path}"
            )
        read = read_events(self._session_path)
        if writer.cut_tail:
            _LOG.warning("cut a torn final line of %d bytes", len(writer.cut_tail))
        events = list(read.events)
        chain = run_events(events, ref.run_id)
        if not chain:
            raise ResumeRefusedError(f"the session has no run {ref.run_id}")
        self._check_digest(chain)
        # With the session lock held: two unlocked resumes would both write the lines.
        for line in repair_spend(
            self._spend_path, chain, run_key(ref.session_id, ref.run_id)
        ):
            _LOG.warning("spend log line not counted: %r", line)
        run = self._new_run(writer, ref.run_id, limits_in_force(chain), events, chain)
        run.state = replay(chain)
        run.next_seq = sum(1 for e in events if e.run_id == ref.run_id)
        return run

    def _check_digest(self, chain: Sequence[Event]) -> None:
        root = next(e for e in chain if isinstance(e, RunStarted))
        if root.config_digest != self._config.digest():
            raise ConfigChangedError(
                "the configuration differs from the one this run started with; "
                "rebuild the same system prompt and tool schemas"
            )
