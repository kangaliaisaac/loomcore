"""The verifier hook: check the model's claim to be done (spec 3.6).

Pure types and the critique text. The driver calls the verifier when the model
claims it is done (a `finish` call or an `end_turn`), writes the verdict, and lets
the run end or sends the critique back to the model.
"""

from collections.abc import Callable
from dataclasses import dataclass

from loomcore.replay import RunState


@dataclass(frozen=True)
class Verdict:
    passed: bool
    verifier: str  # who judged, logged on `verification_passed` / `_failed`
    message: str = ""  # the short critique on a failure; no secret in it


class VerifierError(Exception):
    """The verifier could not judge (its tests would not run, its check timed out).
    The run ends with reason `error`; any other exception is a bug and propagates."""


# It checks the outcome (tests, a checker function), may do I/O, and should be
# idempotent: a crash between the check and its event runs it again. It raises
# `VerifierError` when it cannot reach a verdict, and bounds its own time.
Verifier = Callable[[RunState], Verdict]


def critique(verifier: str, message: str) -> str:
    """What the model is told when a claim is rejected."""
    return f"Verification failed ({verifier}): {message}\nFix this, then finish again."
