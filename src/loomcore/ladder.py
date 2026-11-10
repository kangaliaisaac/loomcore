"""The escalation ladder's pure parts: the repeat detector and the rung messages.

The driver judges a turn's calls once, in emission order, then writes an
`escalation` event; `step` turns the rung into effects (spec section 1).
"""

from collections.abc import Sequence

NUDGE_RUNG = 1
SWITCH_RUNG = 2
ASK_RUNG = 3
STUCK_RUNG = 4

NUDGE = (
    "You have made this same call several times in a row. Do not repeat it: "
    "change the input, use another tool, or finish with what you have."
)
SWITCH = (
    "You are still repeating the same call, so the approach has to change. Follow "
    "any new instructions you have been given, and otherwise use a different tool or "
    "a different input: do not retry what has not worked."
)


def tripped(
    history: Sequence[str], hashes: Sequence[str], window: int, threshold: int
) -> list[int]:
    """Which of this turn's calls trip the detector, as indices into `hashes`.

    `history` is the run's own earlier call hashes. Each call is judged in emission
    order against the last `window` calls up to and including itself, and trips when
    at least `threshold` of them have its hash.
    """
    return [
        k
        for k, call in enumerate(hashes)
        if (*history, *hashes[: k + 1])[-window:].count(call) >= threshold
    ]


def reason(name: str, threshold: int, window: int) -> str:
    return f"at least {threshold} identical {name} calls in the last {window} calls"
