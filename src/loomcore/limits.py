"""Limits: when a run must pause, as a pure check over the state (spec section 1).

The driver reads the month's spend and prices the next call at the edge and passes
both in; nothing here touches the clock, the spend log or a model.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from loomcore.events import (
    Event,
    JsonObject,
    RunForked,
    RunResumed,
    RunStarted,
    Usd,
)
from loomcore.months import month_key
from loomcore.replay import RunState

LimitName = Literal["max_output_tokens", "max_turns", "timeout", "run_cap", "month_cap"]

_TRUNCATIONS_BEFORE_PAUSE = 2  # the first cut-off continues, the second pauses


class Limits(BaseModel):
    """The limits in force for a run. Both caps must be given, or set to None."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_cap: Usd | None
    month_cap: Usd | None
    max_turns: int = Field(50, gt=0)
    timeout: float = Field(600.0, gt=0)  # seconds of active time
    max_output_tokens: int = Field(8192, gt=0)
    repeat_window: int = Field(10, gt=0)
    repeat_threshold: int = Field(3, gt=0)

    def to_json(self) -> JsonObject:
        """The form the log holds, for `run_started`, `run_forked`, `run_resumed`."""
        return self.model_dump(mode="json")


class LimitsMissingError(Exception):
    """The chain holds no event that carries limits; the log is not a run's."""


@dataclass(frozen=True)
class WithinLimits:
    pass


@dataclass(frozen=True)
class LimitReached:
    limit: LimitName


def limits_in_force(chain: Sequence[Event]) -> Limits:
    """The limits for the run `chain` ends in: the latest event that carries some.

    `chain` is `run_events` of that run, so a sibling's override is not in it, and
    a fork sees its parent's events only up to `from.seq`.
    """
    for event in reversed(chain):
        match event:
            case (
                RunResumed(limits=raw) | RunForked(limits=raw) | RunStarted(limits=raw)
            ):
                if raw is not None:
                    return Limits.model_validate(raw)
    raise LimitsMissingError("no run_started, run_forked or run_resumed carries limits")


def elapsed(state: RunState, now: datetime) -> timedelta:
    """Active time: closed segments, plus the open one up to `now`."""
    if state.segment_start is None:
        return state.active
    return state.active + (now - state.segment_start)


def check_limits(
    state: RunState,
    limits: Limits,
    month_to_date: Decimal,
    worst_case: Decimal,
    now: datetime,
) -> WithinLimits | LimitReached:
    """Whether the next model call may be made; the first limit in this order wins."""
    run_spend = sum(state.spend_by_month.values(), Decimal(0))
    # A run that spans a month boundary charges only this month's calls to this month.
    month_spend = state.spend_by_month.get(month_key(now), Decimal(0))
    checks: tuple[tuple[LimitName, bool], ...] = (
        ("max_output_tokens", state.truncations >= _TRUNCATIONS_BEFORE_PAUSE),
        ("max_turns", state.turns >= limits.max_turns),
        ("timeout", elapsed(state, now) >= timedelta(seconds=limits.timeout)),
        ("run_cap", _crosses(limits.run_cap, run_spend + worst_case)),
        (
            "month_cap",
            _crosses(limits.month_cap, month_to_date + month_spend + worst_case),
        ),
    )
    return next(
        (LimitReached(name) for name, hit in checks if hit),
        WithinLimits(),
    )


def _crosses(cap: Decimal | None, total: Decimal) -> bool:
    return cap is not None and total > cap
