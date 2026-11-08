"""Hypothesis strategies for event logs."""

from factories import PAYLOADS, calling, make
from hypothesis import strategies as st

from loomcore.events import Event

TOOL_IDS = st.sampled_from(["a", "b", "c"])


@st.composite
def events_for(draw: st.DrawFn, run: str, seq: int) -> Event:
    """Any kind, in any order: replay must be total over whatever a log holds."""
    kind = draw(st.sampled_from(sorted(PAYLOADS)))
    payload: dict[str, object] = {}
    if kind == "model_call":
        stop = draw(st.sampled_from(["tool_use", "max_tokens", "end_turn"]))
        ids = draw(st.lists(TOOL_IDS, max_size=3, unique=True))
        payload["message"] = calling(*ids, stop_reason=stop)
    elif kind in {
        "tool_completed",
        "approval_requested",
        "approval_granted",
        "approval_denied",
    }:
        payload["tool_use_id"] = draw(TOOL_IDS)
    elif kind == "run_paused":
        payload.update(reason="limit", limit="max_turns")
    elif kind == "run_resumed":
        payload.update(reason="crash")
    elif kind == "run_forked":
        payload["from"] = {"runId": run, "seq": 0}
    return make(kind, seq, run, **payload)


@st.composite
def logs(draw: st.DrawFn, run: str = "r1") -> list[Event]:
    size = draw(st.integers(min_value=0, max_value=25))
    return [draw(events_for(run, seq)) for seq in range(size)]
