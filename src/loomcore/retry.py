"""Retries: a pure policy from the attempt and the error to a delay or a stop.

The random source is passed in, so a test fixes the jitter. The clock and the
sleep belong to the driver (spec section 1, Retries).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Random = Callable[[], float]  # uniform in [0, 1)

_RETRYABLE = frozenset({429, 529})  # rate limited, overloaded
_SERVER_ERRORS = range(500, 600)


@dataclass(frozen=True)
class CallError:
    """What a failed model or tool call tells the policy; no SDK types here."""

    kind: Literal["http", "connection", "timeout"]
    status: int | None = None  # for "http"
    retry_after: float | None = None  # seconds, from the `retry-after` header


@dataclass(frozen=True)
class RetryPolicy:
    base: float = 0.5  # seconds
    cap: float = 30.0
    max_attempts: int = 5  # retries, not counting the first call


@dataclass(frozen=True)
class Delay:
    seconds: float


@dataclass(frozen=True)
class GiveUp:
    reason: str


def retryable(error: CallError) -> bool:
    """429, 529, other 5xx, connection errors and timeouts; no other 4xx."""
    if error.kind != "http":
        return True
    status = error.status
    return status is not None and (status in _RETRYABLE or status in _SERVER_ERRORS)


def decide(
    attempt: int,
    error: CallError,
    random: Random,
    policy: RetryPolicy = RetryPolicy(),  # noqa: B008  (frozen, so shared safely)
) -> Delay | GiveUp:
    """`attempt` is 0 for the first failure. Full jitter, floored by retry-after."""
    if not retryable(error):
        return GiveUp(f"not retryable: {_describe(error)}")
    if attempt >= policy.max_attempts:
        return GiveUp(f"gave up after {policy.max_attempts} retries")
    jittered = random() * min(policy.cap, policy.base * 2**attempt)
    return Delay(max(jittered, error.retry_after or 0.0))


def _describe(error: CallError) -> str:
    return f"HTTP {error.status}" if error.kind == "http" else error.kind
