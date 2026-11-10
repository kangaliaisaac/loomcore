"""f10: the retry policy is pure, honours retry-after and never retries a 4xx."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from loomcore.retry import CallError, Delay, GiveUp, RetryPolicy, decide, retryable


def http(status: int, retry_after: float | None = None) -> CallError:
    return CallError("http", status, retry_after)


def never() -> float:
    raise AssertionError("the random source must not be read for a give-up")


@pytest.mark.parametrize("status", [429, 529, 500, 502, 503, 504, 599])
def test_rate_limits_overload_and_server_errors_are_retried(status: int) -> None:
    assert retryable(http(status))


@pytest.mark.parametrize("kind", ["connection", "timeout"])
def test_connection_errors_and_timeouts_are_retried(kind: str) -> None:
    assert retryable(CallError(kind))  # type: ignore[arg-type]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 409, 413, 422, 499])
def test_other_4xx_is_never_retried(status: int) -> None:
    result = decide(0, http(status), never)
    assert isinstance(result, GiveUp)
    assert f"HTTP {status}" in result.reason


def test_a_status_that_is_not_an_error_is_not_retried() -> None:
    assert not retryable(http(200))
    assert not retryable(CallError("http"))


def test_the_delay_is_full_jitter_over_an_exponential_ceiling() -> None:
    policy = RetryPolicy(base=1.0, cap=100.0, max_attempts=10)
    assert decide(0, http(503), lambda: 0.5, policy) == Delay(0.5)
    assert decide(3, http(503), lambda: 0.5, policy) == Delay(4.0)
    assert decide(3, http(503), lambda: 0.0, policy) == Delay(0.0)


def test_the_ceiling_is_capped() -> None:
    policy = RetryPolicy(base=1.0, cap=8.0, max_attempts=20)
    assert decide(15, http(503), lambda: 1.0, policy) == Delay(8.0)


def test_retry_after_is_a_floor_on_the_delay() -> None:
    policy = RetryPolicy(base=1.0, cap=8.0, max_attempts=5)
    assert decide(0, http(429, retry_after=7.0), lambda: 0.1, policy) == Delay(7.0)
    assert decide(4, http(429, retry_after=1.0), lambda: 1.0, policy) == Delay(8.0)


def test_it_gives_up_after_max_attempts() -> None:
    policy = RetryPolicy(max_attempts=2)
    assert isinstance(decide(1, http(503), lambda: 0.5, policy), Delay)
    result = decide(2, http(503), never, policy)
    assert isinstance(result, GiveUp)
    assert "2 retries" in result.reason


def test_retry_after_above_the_cap_is_honoured_not_capped() -> None:
    policy = RetryPolicy(base=1.0, cap=8.0, max_attempts=5)
    assert decide(0, http(429, retry_after=60.0), lambda: 0.1, policy) == Delay(60.0)


def test_a_huge_attempt_number_is_capped_not_an_overflow() -> None:
    policy = RetryPolicy(base=1.0, cap=8.0, max_attempts=10_000)
    assert decide(5000, http(503), lambda: 1.0, policy) == Delay(8.0)


@given(
    attempt=st.integers(min_value=0, max_value=9),
    draw=st.floats(min_value=0.0, max_value=0.999),
    status=st.sampled_from([429, 500, 503, 529]),
)
def test_a_delay_stays_under_both_the_cap_and_the_exponential_ceiling(
    attempt: int, draw: float, status: int
) -> None:
    policy = RetryPolicy(base=1.0, cap=4.0, max_attempts=10)
    result = decide(attempt, http(status), lambda: draw, policy)
    assert isinstance(result, Delay)
    assert 0.0 <= result.seconds <= min(policy.cap, policy.base * 2**attempt)


def test_the_same_inputs_always_give_the_same_answer() -> None:
    first = decide(2, http(429, retry_after=3.0), lambda: 0.25)
    assert first == decide(2, http(429, retry_after=3.0), lambda: 0.25)
