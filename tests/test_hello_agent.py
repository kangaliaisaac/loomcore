"""The README's hello-agent runs, and the README shows exactly that file."""

import runpy
from collections.abc import Callable
from pathlib import Path

import pytest

import loomcore.anthropic
from loomcore import finish
from loomcore.eventlog import read_events
from loomcore.events import JsonObject
from loomcore.ports import ModelRequest
from loomcore.replay import replay as replay_events

ROOT = Path(__file__).parent.parent
EXAMPLE = ROOT / "examples" / "hello_agent.py"


def reply(n: int, name: str, **args: int | str) -> JsonObject:
    block: JsonObject = {
        "type": "tool_use",
        "id": f"t{n}",
        "name": name,
        "input": dict(args),
    }
    return {
        "id": f"msg_{n}",
        "model": "claude-haiku-5-5",
        "role": "assistant",
        "content": [block],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100 * n, "output_tokens": 20},
    }


def text_reply(n: int, text: str) -> JsonObject:
    return {
        "id": f"msg_{n}",
        "model": "claude-haiku-5-5",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100 * n, "output_tokens": 20},
    }


def scripted(
    replies: list[JsonObject], seen: list[ModelRequest]
) -> Callable[[], object]:
    def call(request: ModelRequest) -> JsonObject:
        seen.append(request)
        return replies.pop(0)

    return lambda: call


def run_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: object) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loomcore.anthropic, "model", model)
    runpy.run_path(str(EXAMPLE))


def test_the_example_runs_a_three_step_task_and_leaves_a_replayable_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[ModelRequest] = []
    replies = [
        reply(1, "multiply", a=6, b=7),
        reply(2, "add", a=42, b=8),
        reply(3, finish.NAME, result="50"),
    ]
    run_example(tmp_path, monkeypatch, scripted(replies, seen))
    assert capsys.readouterr().out.strip() == "finished finished 50"
    events = read_events(tmp_path / "session.jsonl").events
    assert [e.kind for e in events].count("tool_completed") == 3  # two tools and finish
    done = {e.tool_use_id: e for e in events if e.kind == "tool_completed"}
    assert [(done[i].ok, done[i].content) for i in ("t1", "t2")] == [
        (True, "42"),
        (True, "50"),
    ]
    state = replay_events(events)
    assert (state.finished, state.turns) == ("finished", 3)
    # What the model is offered: both tools and finish, and the rule to call finish.
    offered = {str(tool["name"]) for tool in seen[0].tools}
    assert offered == {"multiply", "add", finish.NAME}
    assert "call finish" in seen[0].system


def test_a_model_that_answers_in_plain_text_ends_the_run_with_its_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    replies = [text_reply(1, "6 * 7 + 8 = 50")]
    run_example(tmp_path, monkeypatch, scripted(replies, []))
    assert capsys.readouterr().out.strip() == "finished end_turn 6 * 7 + 8 = 50"


def test_a_run_that_ends_on_an_error_says_so_instead_of_looking_like_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    replies = [{**text_reply(1, "no answer"), "stop_reason": "mystery"}]
    run_example(tmp_path, monkeypatch, scripted(replies, []))
    out = capsys.readouterr().out
    assert out.startswith("finished error ")
    assert "unknown stop_reason: 'mystery'" in out


def test_a_run_that_spends_past_its_cap_pauses_and_names_the_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    costly: JsonObject = {
        **reply(1, "multiply", a=6, b=7),
        "usage": {"input_tokens": 5_000_000, "output_tokens": 20},
    }
    run_example(tmp_path, monkeypatch, scripted([costly], []))
    assert capsys.readouterr().out.strip() == "paused limit run_cap"


def test_the_readme_shows_the_example_file_as_it_is() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("<!-- hello-agent:start -->")
    end = readme.index("<!-- hello-agent:end -->")
    block = readme[start:end].split("```python\n", 1)[1].rsplit("```", 1)[0]
    assert block == EXAMPLE.read_text(encoding="utf-8")
