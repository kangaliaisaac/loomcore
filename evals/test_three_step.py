"""The scripted three-step eval: the whole stack, the model faked, a replayable log."""

import shutil
from decimal import Decimal
from pathlib import Path

import pytest
from three_step import (
    ANSWER,
    ScriptedModel,
    canned_responses,
    driver_on,
    run_eval,
)

from loomcore.conversation import build_messages
from loomcore.eventlog import read_events
from loomcore.events import Event, ModelCall, ToolCompleted, ToolRequested
from loomcore.observability.cost import Tokens, tally, vendored_table
from loomcore.observability.trace import spans
from loomcore.ports import RunRef
from loomcore.replay import replay


@pytest.fixture
def events(tmp_path: Path) -> tuple[Event, ...]:
    ran = run_eval(tmp_path)
    assert ran.result.status == "finished"
    return read_events(ran.session).events


def test_the_run_finishes_with_the_figure_and_writes_the_cited_answer(
    tmp_path: Path,
) -> None:
    ran = run_eval(tmp_path)
    assert (ran.result.status, ran.result.reason, ran.result.result) == (
        "finished",
        "finished",
        "12.5%",
    )
    work = tmp_path / "work"
    written = (work / "answer.md").read_text("utf-8")
    assert written == ANSWER
    figures = dict(
        line.split(": ")
        for line in (work / "filing.txt").read_text("utf-8").splitlines()[2:]
    )  # the two lines the answer cites
    margin = int(figures["Net income"]) / int(figures["Revenue"].replace(",", ""))
    assert f"{margin:.1%}" in written  # the figure the run wrote is the right one


def test_the_log_is_complete(events: tuple[Event, ...]) -> None:
    kinds = [e.kind for e in events]
    assert kinds[0] == "run_started" and kinds[-1] == "run_finished"
    assert [e.seq for e in events] == list(range(len(events)))  # no gap
    requested = [e.tool_use_id for e in events if isinstance(e, ToolRequested)]
    completed = [e.tool_use_id for e in events if isinstance(e, ToolCompleted)]
    assert sorted(requested) == ["t1", "t2"]  # `finish` is not a requested tool
    assert sorted(completed) == ["t1", "t2", "t3"]
    assert len(completed) == len(set(completed))  # exactly one each
    assert kinds.count("model_call") == 3


def test_the_log_replays_to_the_end_state_however_it_is_read(
    tmp_path: Path, events: tuple[Event, ...]
) -> None:
    state = replay(events)
    assert state.finished == "finished"
    again = read_events(tmp_path / "session.jsonl").events  # a second read of the file
    assert replay(again) == state
    assert replay(events[: len(events) // 2]) != state  # a cut gives an earlier state


def test_every_request_the_model_saw_is_rebuilt_from_the_log(tmp_path: Path) -> None:
    ran = run_eval(tmp_path)
    logged = read_events(ran.session).events
    calls = [e for e in logged if isinstance(e, ModelCall)]
    assert len(calls) == len(ran.model.requests) == 3
    for call, request in zip(calls, ran.model.requests, strict=True):
        before = [e for e in logged if e.seq < call.seq]
        assert build_messages(ran.task.prompt, before) == request.messages


def test_a_log_cut_after_step_two_resumes_to_the_same_result(tmp_path: Path) -> None:
    whole = run_eval(tmp_path / "whole")
    lines = whole.session.read_text("utf-8").splitlines(keepends=True)
    logged = read_events(whole.session).events
    last_write = max(
        e.seq for e in logged if isinstance(e, ToolCompleted) and e.tool_use_id == "t2"
    )
    cut = tmp_path / "cut"
    shutil.copytree(tmp_path / "whole" / "work", cut / "work")  # what disk kept
    (cut / "session.jsonl").write_text("".join(lines[: last_write + 1]), "utf-8")
    model = ScriptedModel(script=canned_responses()[2:])  # only the last step is left
    driver, _task = driver_on(cut, model)
    resumed = driver.resume(RunRef("session", "run-1"))
    assert (resumed.status, resumed.result) == ("finished", "12.5%")
    after = read_events(cut / "session.jsonl").events
    assert after[-1].kind == "run_finished"
    assert [e.tool_use_id for e in after if isinstance(e, ToolCompleted)] == [
        "t1",
        "t2",
        "t3",
    ]  # the resume only added the last
    assert [r.messages for r in model.requests] == [whole.model.requests[2].messages]


def test_the_cost_is_the_hand_computed_one_and_the_spans_are_stable(
    tmp_path: Path, events: tuple[Event, ...]
) -> None:
    # Sonnet 5.5 per million tokens: input 2, output 10, cache read 0.10.
    by_hand = (
        Decimal(1000 * 2 + 100 * 10)
        + Decimal(1500 * 2 + 150 * 10)
        + Decimal("0.10") * 1000
        + Decimal(2000 * 2 + 80 * 10)
        + Decimal("0.10") * 2500
    ) / 1_000_000
    run = tally(events, vendored_table())["run-1"]
    assert run.usd == by_hand == Decimal("0.01265")
    assert run.tokens == Tokens(input=4500, output=330, cache_read=3500)
    assert run.calls == 3 and run.unpriced == 0
    assert spans(events) == spans(read_events(tmp_path / "session.jsonl").events)
    kinds = [s.kind for s in spans(events)]
    assert kinds.count("model_call") == 3 and kinds.count("tool_call") == 2
