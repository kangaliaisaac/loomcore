"""f16, part 2: the window is the store applied to the log, turn after turn."""

import json
from collections.abc import Sequence

from factories import calling, make

from loomcore.conversation import build_messages
from loomcore.events import Event, JsonObject
from loomcore.replay import run_events
from loomcore.window import Budget, Carried, Window, build_window


def size(messages: Sequence[JsonObject]) -> int:
    return len(json.dumps(messages, sort_keys=True)) // 4 + 1


def history(
    turns: int, run: str = "r1", start: int = 1, tag: str = "result"
) -> list[Event]:
    """`turns` exchanges, each a tool call with a long result."""
    log: list[Event] = []
    for k in range(turns):
        seq = start + 2 * k
        tool = f"{tag}-{k}"  # tool_use ids are unique across a conversation
        log.append(make("model_call", seq, run, message=calling(tool)))
        log.append(
            make(
                "tool_completed",
                seq + 1,
                run,
                tool_use_id=tool,
                content=(f"{tag}{k} ") * 30,
                ok=True,
            )
        )
    return log


def window(events: list[Event], carried: Carried, tokens: int | None = 600) -> Window:
    budget = None if tokens is None else Budget(tokens, 0.8, 0.6)
    return build_window("go", events, carried, budget, size)


def test_with_no_budget_the_window_is_the_whole_history() -> None:
    log = history(5)
    out = window(log, Carried(), tokens=None)
    assert out.messages == build_messages("go", log)
    assert (out.packed, out.newly_dropped, out.newly_cleared) == (None, (), ())


def test_a_budget_that_holds_everything_changes_nothing() -> None:
    log = history(3)
    out = window(log, Carried(), tokens=100_000)
    assert out.messages == build_messages("go", log)
    assert (out.newly_dropped, out.newly_cleared) == ((), ())


def play(turns: int, tokens: int) -> list[tuple[Window, list[Event]]]:
    """The driver's loop: each turn adds an exchange, builds the window and carries
    forward what it newly dropped and cleared."""
    dropped: list[str] = []
    cleared: list[str] = []
    out: list[tuple[Window, list[Event]]] = []
    for n in range(1, turns + 1):
        log = history(n)
        w = window(log, Carried(tuple(dropped), tuple(cleared)), tokens)
        dropped += w.newly_dropped
        cleared += w.newly_cleared
        out.append((w, log))
    return out


def test_between_two_changes_the_new_window_extends_the_old_one() -> None:
    turns = play(14, 600)
    changes = [bool(w.newly_dropped or w.newly_cleared) for w, _ in turns]
    assert any(changes)  # the budget is small enough to force compaction or drops
    assert not all(changes)  # and not so small that every turn rewrites the window
    for (before, _), (after, _), changed in zip(
        turns, turns[1:], changes[1:], strict=False
    ):
        if not changed:
            assert after.messages[: len(before.messages)] == before.messages


def test_the_task_and_the_latest_exchange_are_always_in_the_window() -> None:
    for out, log in play(14, 600):
        assert out.messages[0] == {"role": "user", "content": "go"}
        assert out.messages[-2:] == build_messages("go", log)[-2:]  # never cleared


def test_a_resume_starts_from_the_carried_sets_and_finds_the_same_window() -> None:
    turns = play(14, 600)
    dropped = [i for w, _ in turns for i in w.newly_dropped]
    cleared = [i for w, _ in turns for i in w.newly_cleared]
    last, log = turns[-1]
    again = window(log, Carried(tuple(dropped), tuple(cleared)))
    assert again.messages == last.messages
    assert (again.newly_dropped, again.newly_cleared) == ((), ())


def test_an_over_window_is_sent_as_it_is_and_records_nothing() -> None:
    log = history(4)
    out = window(log, Carried(), tokens=5)
    assert out.packed is not None
    assert out.packed.over
    assert (out.newly_dropped, out.newly_cleared) == ((), ())
    assert len(out.messages) == 3  # the task and the latest exchange, nothing else


def test_a_fork_starts_without_what_its_parent_had_dropped() -> None:
    first = [make("run_started", 0, "r1"), *history(8, "r1", 1, "parent")]
    fork_at = first[-1].seq
    origin: dict[str, object] = {"from": {"runId": "r1", "seq": fork_at}}
    second = [
        make("run_forked", 0, "r2", hint="b", **origin),
        *history(2, "r2", 1, "fork"),
    ]
    chain = run_events([*first, *second], "r2")
    gone = ("seq:r1:1", "seq:r1:3")  # r1's windows dropped its first two exchanges
    texts = json.dumps(window(chain, Carried(dropped=gone), tokens=100_000).messages)
    assert "parent0 " not in texts
    assert "parent1 " not in texts
    assert "parent2 " in texts
    assert "fork0 " in texts
    whole = json.dumps(window(chain, Carried(), tokens=100_000).messages)
    assert "parent0 " in whole
