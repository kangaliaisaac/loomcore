"""Limits: when a run must pause, as a pure check over the state (spec section 1).

The driver reads the month's spend and prices the next call at the edge and passes
both in; nothing here touches the clock, the spend log or a model.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from loomcore.events import (
    Event,
    JsonObject,
    RunForked,
    RunResumed,
    RunStarted,
    Usd,
)
from loomcore.months import month_key
from loomcore.replay import RunState, open_segment, spent, spent_in

LimitName = Literal["max_output_tokens", "max_turns", "timeout", "run_cap", "month_cap"]

_TRUNCATIONS_BEFORE_PAUSE = 2  # the first cut-off continues, the second pauses


class Limits(BaseModel):
    """The limits in force for a run. Both caps must be given, or set to None."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_cap: Annotated[Usd, Field(ge=0)] | None
    month_cap: Annotated[Usd, Field(ge=0)] | None
    max_turns: int = Field(50, gt=0)
    timeout: float = Field(600.0, gt=0)  # seconds of active time
    max_output_tokens: int = Field(8192, gt=0)
    repeat_window: int = Field(10, gt=0)
    repeat_threshold: int = Field(3, gt=0)
    # The context window (spec 3.5): None means no packing, and requests carry the
    # whole history. `compact_at` and `compact_to` are fractions of the budget.
    context_budget: int | None = Field(None, gt=0)
    compact_at: float = Field(0.8, gt=0, le=1)
    compact_to: float = Field(0.6, gt=0, lt=1)

    @model_validator(mode="after")
    def _compaction_has_room(self) -> Self:
        if self.compact_to >= self.compact_at:
            raise ValueError(
                "compact_to must be below compact_at, or it never gains room"
            )
        return self

    def to_json(self) -> JsonObject:
        """The form the log holds, for `run_started`, `run_forked`, `run_resumed`."""
        return self.model_dump(mode="json")


class LimitsError(Exception):
    """The chain has no usable limits: none carried, or the latest do not parse."""


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
                    return _parse(raw, event.seq)
    raise LimitsError("no run_started, run_forked or run_resumed carries limits")


def _parse(raw: JsonObject, seq: int) -> Limits:
    try:
        return Limits.model_validate(raw)
    except ValidationError as error:
        raise LimitsError(
            f"the limits at seq {seq} do not parse; write limits from Limits.to_json()"
        ) from error


def elapsed(state: RunState, now: datetime) -> timedelta:
    """Active time: closed segments, plus the open one.

    A paused or finished run is not running, so its open segment ends at its last
    event, not at `now`.
    """
    stopped = state.paused or state.finished is not None
    return state.active + open_segment(state, state.last_event_at if stopped else now)


def check_limits(
    state: RunState,
    limits: Limits,
    month_to_date: Decimal,
    worst_case: Decimal,
    now: datetime,
) -> WithinLimits | LimitReached:
    """Whether the next model call may be made; the first limit in this order wins."""
    run_spend = spent(state)
    # A run that spans a month boundary charges only this month's calls to this month.
    month_spend = spent_in(state, month_key(now))
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
