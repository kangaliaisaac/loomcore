"""The UTC month, the one definition the spend log and the run state share."""

from datetime import UTC, datetime


def month_key(moment: datetime) -> str:
    """The UTC month of `moment`, like "2026-11"."""
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware: months are judged in UTC")
    return f"{moment.astimezone(UTC):%Y-%m}"
