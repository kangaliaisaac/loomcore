"""The driver: run, resume and fork a run (spec section 1).

It opens the session under its lock, checks that a resume is allowed and that the
configuration matches, repairs the spend log, and hands the run to `engine.Run`.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from loomcore.engine import ResumeRefusedError, Run
from loomcore.eventlog import EventWriter, open_writer, read_events
from loomcore.events import Escalation, Event, ForkOrigin, RunStarted
from loomcore.limits import Limits, limits_in_force
from loomcore.ports import Config, Decision, Ports, RunRef, RunResult, Task
from loomcore.replay import replay, run_events
from loomcore.retry import RetryPolicy
from loomcore.rewind import ensure_fork_point
from loomcore.spend import repair_spend, run_key

_LOG = logging.getLogger(__name__)


class ConfigChangedError(Exception):
    """The configuration this process holds is not the one the run started with."""


class RunIdTakenError(Exception):
    """The `new_run_id` port returned an id the session file already uses."""


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
            _note_cut(writer)
            taken = read_events(self._session_path).events
            run = self._new_run(
                writer, self._fresh_id(taken), task.limits, [], self._config
            )
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
        self._require_session()
        with open_writer(self._session_path) as writer:
            run = self._reopen(writer, ref)
            return run.drive(run.reopen(decision, limits))

    def fork(
        self, origin: ForkOrigin, hint: str | None = None, limits: Limits | None = None
    ) -> RunResult:
        self._require_session()
        with open_writer(self._session_path) as writer:
            _note_cut(writer)
            events = list(read_events(self._session_path).events)
            ensure_fork_point(events, origin)
            chain = run_events(events, origin.run_id, upto=origin.seq)
            run = self._new_run(
                writer,
                self._fresh_id(events),
                limits_in_force(chain),
                chain,
                self._config_in_force(chain),
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
        chain: list[Event],
        config: Config,
    ) -> Run:
        ref = RunRef(self._session_path.stem, run_id)
        return Run(
            config=config,
            switch_to=self._config.switch_to,
            ports=self._ports,
            retry=self._retry,
            spend_path=self._spend_path,
            writer=writer,
            ref=ref,
            limits=limits,
            chain=chain,
        )

    def _reopen(self, writer: EventWriter, ref: RunRef) -> Run:
        if ref.session_id != self._session_path.stem:
            raise ResumeRefusedError(
                f"{ref.session_id} is not the session in {self._session_path}"
            )
        _note_cut(writer)
        events = list(read_events(self._session_path).events)
        chain = run_events(events, ref.run_id)
        if not chain:
            raise ResumeRefusedError(f"the session has no run {ref.run_id}")
        config = self._config_in_force(chain)
        # With the session lock held: two unlocked resumes would both write the lines.
        for line in repair_spend(
            self._spend_path, chain, run_key(ref.session_id, ref.run_id)
        ):
            _LOG.warning("spend log line not counted: %r", line)
        run = self._new_run(writer, ref.run_id, limits_in_force(chain), chain, config)
        run.state = replay(chain)
        run.next_seq = sum(1 for e in events if e.run_id == ref.run_id)
        return run

    def _fresh_id(self, events: Sequence[Event]) -> str:
        """A new run id; run ids are unique within the session file."""
        run_id = self._ports.new_run_id()
        if any(e.run_id == run_id for e in events):
            raise RunIdTakenError(f"run id {run_id} is already in the session file")
        return run_id

    def _require_session(self) -> None:
        """Refuse before the lock is taken: taking it would create the file."""
        if not self._session_path.exists():
            raise ResumeRefusedError(f"there is no session file {self._session_path}")

    def _config_in_force(self, chain: Sequence[Event]) -> Config:
        """The configuration this process must hold for the run `chain` ends in.

        The root `run_started` fixes the base; the latest switch-strategy
        `escalation` fixes the one in force. Either mismatch raises.
        """
        root = next(e for e in chain if isinstance(e, RunStarted))
        if root.config_digest != self._config.digest():
            raise ConfigChangedError(_CHANGED)
        switched = next(
            (
                e
                for e in reversed(chain)
                if isinstance(e, Escalation) and e.config_digest
            ),
            None,
        )
        if switched is None:
            return self._config
        target = self._config.switch_to
        if target is None or target.digest() != switched.config_digest:
            raise ConfigChangedError(_CHANGED)
        return target


_CHANGED = (
    "the configuration differs from the one this run is using; rebuild the same "
    "model, system prompt and tool schemas, and the same switch-strategy configuration"
)


def _note_cut(writer: EventWriter) -> None:
    if writer.cut_tail:
        _LOG.warning("cut a torn final line of %d bytes", len(writer.cut_tail))
