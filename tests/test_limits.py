"""f09: limits pause the run by name, and a resume with limits continues it."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from factories import START, calling, make

from loomcore.events import Event, JsonObject
from loomcore.limits import (
    LimitReached,
    Limits,
    LimitsMissingError,
    WithinLimits,
    check_limits,
    limits_in_force,
)
from loomcore.replay import RunState, replay, run_events

CAPS: JsonObject = {"run_cap": None, "month_cap": None}
NO_MONEY = Decimal(0)


def limits(**changes: object) -> Limits:
    return Limits.model_validate({**CAPS, **changes})


def check(
    events: list[Event],
    given: Limits,
    *,
    month_to_date: str = "0",
    worst_case: str = "0",
    after: float = 0,
) -> WithinLimits | LimitReached:
    now = events[-1].timestamp + timedelta(seconds=after)
    state = replay(events)
    return check_limits(state, given, Decimal(month_to_date), Decimal(worst_case), now)


def started(**fields: object) -> Event:
    return make("run_started", 0, limits={**CAPS, **fields})


def call(seq: int, cost: float = 0.01, stop: str = "end_turn") -> Event:
    message = calling(stop_reason=stop)
    return make("model_call", seq, message=message, cost_usd=cost)


def turns(count: int) -> list[Event]:
    return [make("turn_started", n, turn=n) for n in range(1, count + 1)]


def test_a_run_inside_every_limit_may_call_the_model() -> None:
    assert check([started()], limits()) == WithinLimits()


def test_both_caps_must_be_given_or_set_to_none() -> None:
    with pytest.raises(ValueError, match="run_cap"):
        Limits.model_validate({"month_cap": None})


def test_the_defaults_are_the_specs() -> None:
    given = limits()
    assert (given.max_turns, given.timeout, given.max_output_tokens) == (50, 600, 8192)
    assert (given.repeat_window, given.repeat_threshold) == (10, 3)


def test_limits_round_trip_through_the_log() -> None:
    given = limits(run_cap=1.5, max_turns=7)
    assert Limits.model_validate(given.to_json()) == given


def test_max_turns_pauses_when_the_turns_are_used() -> None:
    log = [started(), *turns(2)]
    assert check(log, limits(max_turns=3)) == WithinLimits()
    assert check(log, limits(max_turns=2)) == LimitReached("max_turns")


def test_timeout_pauses_on_active_time() -> None:
    log = [started()]
    assert check(log, limits(timeout=60), after=59) == WithinLimits()
    assert check(log, limits(timeout=60), after=60) == LimitReached("timeout")


def test_run_cap_counts_the_next_calls_worst_case() -> None:
    log = [started(), call(1, 0.4)]
    cap = limits(run_cap=1)
    assert check(log, cap, worst_case="0.6") == WithinLimits()
    assert check(log, cap, worst_case="0.61") == LimitReached("run_cap")


def test_month_cap_counts_what_other_runs_spent_this_month() -> None:
    log = [started(), call(1, 0.4)]
    cap = limits(month_cap=10)
    assert check(log, cap, month_to_date="9.6") == WithinLimits()
    assert check(log, cap, month_to_date="9.7") == LimitReached("month_cap")


def test_month_cap_leaves_out_this_runs_spend_from_an_earlier_month() -> None:
    old = make("model_call", 1, message=calling(), cost_usd=5)
    new_month = START.replace(month=12)
    assert old.timestamp.month == 11
    state = replay([started(), old])
    now = new_month.astimezone(UTC)
    within = check_limits(
        state, limits(month_cap=1, timeout=1e9), Decimal(0), Decimal(0), now
    )
    over_run = check_limits(
        state, limits(run_cap=1, timeout=1e9), Decimal(0), Decimal(0), now
    )
    assert within == WithinLimits()  # November's $5 is not December's
    assert over_run == LimitReached("run_cap")  # but it is this run's


def test_a_naive_now_is_refused() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        check_limits(RunState(), limits(), NO_MONEY, NO_MONEY, datetime(2026, 11, 3))


def test_a_cap_of_none_is_unlimited() -> None:
    log = [started(), call(1, 1_000_000)]
    assert check(log, limits(), month_to_date="1e9", worst_case="1e9") == WithinLimits()


def test_the_second_truncated_call_in_a_row_pauses() -> None:
    cut = [started(), call(1, stop="max_tokens")]
    assert check(cut, limits()) == WithinLimits()
    twice = [*cut, call(2, stop="max_tokens")]
    assert check(twice, limits()) == LimitReached("max_output_tokens")


def test_any_other_call_resets_the_truncation_streak() -> None:
    log = [started(), call(1, stop="max_tokens"), call(2), call(3, stop="max_tokens")]
    assert check(log, limits()) == WithinLimits()


def test_a_fork_starts_with_no_truncations_and_no_spend() -> None:
    parent = [started(), call(1, 9, "max_tokens"), call(2, 9, "max_tokens")]
    log = run_events([*parent, forked(3, "r2", parent_seq=2)], "r2")
    assert replay(log).truncations == 0  # the parent's streak is not inherited
    assert check(log, limits(run_cap=1)) == WithinLimits()


def forked(seq: int, run: str, parent_seq: int = 0, **fields: object) -> Event:
    origin = {"runId": "r1", "seq": parent_seq}
    return make("run_forked", seq, run, **{"from": origin, **fields})


def paused(seq: int, limit: str) -> Event:
    return make("run_paused", seq, reason="limit", limit=limit, resume_from=seq - 1)


def resumed(seq: int, reason: str = "limit", limits: JsonObject | None = None) -> Event:
    return make("run_resumed", seq, reason=reason, resume_from=seq - 1, limits=limits)


def test_resuming_after_a_truncation_pause_gives_fresh_attempts() -> None:
    log = [started(), call(1, stop="max_tokens"), call(2, stop="max_tokens")]
    log += [paused(3, "max_output_tokens"), resumed(4, limits=limits().to_json())]
    assert check(log, limits()) == WithinLimits()
    log.append(call(5, stop="max_tokens"))
    assert check(log, limits()) == WithinLimits()  # one cut-off continues again


def test_a_crash_resume_keeps_the_truncation_streak() -> None:
    log = [started(), call(1, stop="max_tokens"), call(2, stop="max_tokens")]
    log.append(resumed(3, "crash"))
    assert check(log, limits()) == LimitReached("max_output_tokens")


def test_resuming_without_raising_a_limit_pauses_again_with_its_name() -> None:
    log = [started(), *turns(2), paused(3, "max_turns")]
    log.append(resumed(4, limits=limits(max_turns=2).to_json()))
    assert check(log, limits(max_turns=2)) == LimitReached("max_turns")
    assert check(log, limits(max_turns=3)) == WithinLimits()


def test_the_turn_count_and_spend_carry_over_a_resume() -> None:
    log = [started(), *turns(2), call(3, 0.4), paused(4, "max_turns")]
    log.append(resumed(5, limits=limits(run_cap=1, max_turns=5).to_json()))
    state = replay(log)
    assert state.turns == 2
    assert sum(state.spend_by_month.values()) == Decimal("0.4")


def test_a_pause_waiting_for_approval_does_not_count_against_the_timeout() -> None:
    hour = 3600
    log = [
        started(),
        make("run_paused", 30, reason="awaiting_approval", resume_from=0),
        make("run_resumed", hour, reason="approval", resume_from=0),
    ]
    assert check(log, limits(timeout=60), after=10) == WithinLimits()
    assert check(log, limits(timeout=60), after=31) == LimitReached("timeout")


def test_a_crash_then_a_late_resume_does_not_time_out() -> None:
    day = 86400
    log = [started(), call(40), make("run_resumed", day, reason="crash", resume_from=1)]
    assert check(log, limits(timeout=60), after=10) == WithinLimits()
    assert check(log, limits(timeout=60), after=21) == LimitReached("timeout")


def test_a_forks_clock_starts_at_its_run_forked() -> None:
    parent = [started(), call(500)]
    log = run_events([*parent, forked(600, "r2")], "r2")
    assert check(log, limits(timeout=60), after=59) == WithinLimits()


def test_the_limits_in_force_are_the_latest_resume_else_fork_else_start() -> None:
    first = limits(max_turns=5)
    later = limits(max_turns=9)
    start = started(max_turns=5)
    assert limits_in_force([start]) == first
    assert limits_in_force([start, resumed(1, limits=later.to_json())]) == later


def test_a_forks_limits_come_from_its_parent_chain_unless_it_overrides() -> None:
    parent = [started(max_turns=5), resumed(1, limits=limits(max_turns=8).to_json())]
    origin = {"from": {"runId": "r1", "seq": 1}}
    plain = make("run_forked", 0, "r2", **origin)
    assert limits_in_force(run_events([*parent, plain], "r2")).max_turns == 8
    override = make(
        "run_forked", 0, "r2", limits=limits(max_turns=2).to_json(), **origin
    )
    assert limits_in_force(run_events([*parent, override], "r2")).max_turns == 2


def test_a_siblings_override_never_applies() -> None:
    parent = [started(max_turns=5)]
    origin = {"from": {"runId": "r1", "seq": 0}}
    mine = make("run_forked", 0, "r2", **origin)
    sibling = make(
        "run_forked", 0, "r3", limits=limits(max_turns=1).to_json(), **origin
    )
    assert limits_in_force(run_events([*parent, mine, sibling], "r2")).max_turns == 5


def test_a_chain_with_no_limits_is_refused() -> None:
    with pytest.raises(LimitsMissingError):
        limits_in_force([make("turn_started", 0)])


def test_when_several_limits_hold_the_first_in_the_specs_order_is_named() -> None:
    log = [started(), *turns(3), call(4, 5)]
    both = limits(max_turns=3, run_cap=1, month_cap=1)
    assert check(log, both) == LimitReached("max_turns")
    caps = limits(run_cap=1, month_cap=1)
    assert check(log, caps, month_to_date="9") == LimitReached("run_cap")


def test_an_approval_resume_keeps_the_truncation_streak() -> None:
    log = [started(), call(1, stop="max_tokens"), call(2, stop="max_tokens")]
    log += [make("run_paused", 3, reason="awaiting_approval", resume_from=2)]
    log += [resumed(4, "approval")]
    assert check(log, limits()) == LimitReached("max_output_tokens")
