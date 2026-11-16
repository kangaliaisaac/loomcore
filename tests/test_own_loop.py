"""f24: the components work without the driver (spec section 1, "Extension seams").

`run_loop` is a hand-written loop of its own. It calls the tool registry, the guard
pipelines, the sandbox's rule evaluation, the workspace and the context window
directly, with a scripted model, and completes a task. It imports nothing of the
driver (`loop`, `engine`, `step`): a test parses this file to check it.
"""

import ast
import itertools
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from loomcore import finish
from loomcore.compose import ToolKit, tool_kit
from loomcore.conversation import call_hash
from loomcore.events import Event, JsonObject, parse_event
from loomcore.guards import (
    Allow,
    Ask,
    Deny,
    Guards,
    PostDecision,
    PreDecision,
    Redact,
    ToolRequest,
    run_post,
    run_pre,
)
from loomcore.ports import ToolResult
from loomcore.replay import RunState, content_blocks, replay, tool_use_ids
from loomcore.sandbox import Refusal, Request, Rules, evaluate
from loomcore.window import Carried, build_window
from loomcore.workspace import Workspace

SECRET = "not-a-real-key"  # a stand-in; a real shape would trip the secret scan
FILING = f"Revenue: 2,000\nNet income: 250\nContact key: {SECRET}\n"
FORBIDDEN = {"loomcore.loop", "loomcore.engine", "loomcore.step"}
TASK = "Read filing.txt and write the net margin to answer.md, then finish."


def read_file(workspace: Workspace, path: str) -> str:
    """Read a text file in the workspace."""
    return workspace.read(path)


def write_file(workspace: Workspace, path: str, content: str) -> str:
    """Write a text file in the workspace."""
    workspace.write(path, content)
    return f"wrote {path}"


@dataclass(frozen=True)
class KeepFiling:
    """A pre guard: the source document is never overwritten."""

    def check(self, request: ToolRequest, state: RunState) -> PreDecision:
        if request.name == "write_file" and request.input.get("path") == "filing.txt":
            return Deny("filing.txt is the source; write your answer elsewhere")
        return Allow()


@dataclass(frozen=True)
class AskOutsideAnswer:
    """A pre guard that wants a human's yes before any other file is written."""

    def check(self, request: ToolRequest, state: RunState) -> PreDecision:
        path = request.input.get("path")
        if request.name == "write_file" and path not in {"answer.md", "filing.txt"}:
            return Ask(f"write {path}?")
        return Allow()


@dataclass(frozen=True)
class HideKeys:
    """A post guard: a key never reaches the model."""

    def check(
        self, request: ToolRequest, result: ToolResult, state: RunState
    ) -> PostDecision:
        if isinstance(result.content, str) and SECRET in result.content:
            return Redact(result.content.replace(SECRET, "[key]"), "an API key")
        return Allow()


def say(call_id: str, name: str, **args: JsonValue) -> JsonObject:
    """A scripted model response: a line of text, then one tool call."""
    text: JsonObject = {"type": "text", "text": f"Calling {name}."}
    block: JsonObject = {"type": "tool_use", "id": call_id, "name": name, "input": args}
    return {"content": [text, block], "stop_reason": "tool_use"}


def log(kind: str, seq: int, **fields: JsonValue) -> Event:
    """One event, with the envelope a log line carries."""
    envelope: JsonObject = {
        "schema": "loomcore.event.v1",
        "seq": seq,
        "timestamp": "2026-11-20T09:00:00.000+00:00",
        "sessionId": "own",
        "runId": "r",
        "kind": kind,
    }
    return parse_event(json.dumps(envelope | fields))


def act(kit: ToolKit, rules: Rules, guards: Guards, events: list[Event]) -> ToolResult:
    """Judge, run and screen the call the model made last; the guards see the state
    the log so far folds to. This loop has no human to ask, so an `Ask` is a denial."""
    last = events[-1]
    assert last.kind == "tool_requested"
    request = ToolRequest(last.tool_use_id, last.name, last.input, last.call_hash)
    verdict = run_pre(guards, request, replay(events))
    if isinstance(verdict, Ask):
        result = ToolResult(False, "approval is not supported here", from_tool=False)
    elif isinstance(verdict, Deny):
        result = ToolResult(False, verdict.reason, from_tool=False)
    elif request.name == "run":  # the sandbox's rules; this loop has no container
        command = tuple(str(part) for part in request.input["command"])  # type: ignore[union-attr]
        refused = evaluate(rules, Request(command, rules.root, frozenset()))
        reason = (
            refused.reason
            if isinstance(refused, Refusal)
            else "allowed by the rules, but this loop has no container to run it in"
        )
        result = ToolResult(False, reason, from_tool=False)
    else:
        result = kit.registry.execute(request.name, request.input)
    return run_post(guards, request, result, replay(events)).result


def run_loop(
    root: Path, model: Callable[[list[JsonObject]], JsonObject]
) -> tuple[JsonValue, list[Event]]:
    kit = tool_kit(root)
    kit.registry.register(read_file, read_only=True)
    kit.registry.register(write_file, writes=("answer.md",))
    rules = Rules(root=root, image="img", commands=frozenset({"/bin/cat"}))
    guards = Guards(pre=(KeepFiling(), AskOutsideAnswer()), post=(HideKeys(),))
    events: list[Event] = []
    for _turn in range(10):
        window = build_window(TASK, events, Carried(), None, lambda _m: 0)
        message = model(window.messages)
        events.append(
            log(
                "model_call",
                len(events),
                message=message,
                duration_ms=0,
                cost_usd=0,
                pricing_version="none",
            )
        )
        calls = set(tool_use_ids(message))  # a call cut off by max_tokens is not one
        for block in (b for b in content_blocks(message) if b.get("id") in calls):
            args = block["input"]
            assert isinstance(args, dict)
            name, use_id = str(block["name"]), str(block["id"])
            if name == finish.NAME:
                return finish.result_of(args), events
            events.append(
                log(
                    "tool_requested",
                    len(events),
                    tool_use_id=use_id,
                    name=name,
                    input=args,
                    call_hash=call_hash(name, args),
                )
            )
            done = act(kit, rules, guards, events)
            events.append(
                log(
                    "tool_completed",
                    len(events),
                    tool_use_id=use_id,
                    ok=done.ok,
                    content=done.content,
                    duration_ms=0,
                    cached=False,
                    from_tool=done.from_tool,
                )
            )
    raise RuntimeError("the model never finished")


def cut_off(call_id: str, name: str, **args: JsonValue) -> JsonObject:
    """A response `max_tokens` cut off in the middle of its tool call."""
    return say(call_id, name, **args) | {"stop_reason": "max_tokens"}


SCRIPT = [
    say("c1", "run", command=["cat", "filing.txt"]),  # a name, not an absolute path
    say("c2", "read_file", path="filing.txt"),
    say("c3", "write_file", path="filing.txt", content="oops"),  # the guard denies it
    say("c3b", "write_file", path="scratch.txt", content="x"),  # a guard asks: no human
    cut_off("c3c", "write_file", path="answer.md", content="12.5"),  # never runs
    say("c4", "write_file", path="answer.md", content="12.5% (net income / revenue)"),
    say("c5", finish.NAME, result="12.5%"),
]


def scripted() -> tuple[
    Callable[[list[JsonObject]], JsonObject], list[list[JsonObject]]
]:
    script, seen = list(SCRIPT), []

    def model(messages: list[JsonObject]) -> JsonObject:
        seen.append(messages)
        return script.pop(0)

    return model, seen


def test_a_hand_written_loop_completes_a_task_with_no_driver(tmp_path: Path) -> None:
    (tmp_path / "filing.txt").write_text(FILING, "utf-8")
    model, seen = scripted()
    result, events = run_loop(tmp_path, model)
    assert result == "12.5%"
    assert (tmp_path / "answer.md").read_text("utf-8").startswith("12.5%")
    assert (tmp_path / "filing.txt").read_text("utf-8") == FILING  # the guard held
    done = {e.tool_use_id: e for e in events if e.kind == "tool_completed"}
    assert not done["c1"].ok and "absolute path" in str(done["c1"].content)  # sandbox
    assert not done["c3"].ok and "the source" in str(done["c3"].content)  # pre guard
    assert not done["c3b"].ok and "approval is not supported" in str(
        done["c3b"].content
    )
    assert not (tmp_path / "scratch.txt").exists()  # an Ask is not an Allow
    assert "c3c" not in done and "c3c" not in {  # the cut-off call is not a call
        e.tool_use_id for e in events if e.kind == "tool_requested"
    }
    assert (tmp_path / "answer.md").read_text("utf-8").startswith("12.5%")
    assert done["c2"].ok and SECRET not in str(done["c2"].content)  # post guard
    assert "[key]" in str(done["c2"].content)
    assert SECRET not in json.dumps(seen)  # the model never saw it
    assert len(seen) == len(SCRIPT)  # one window per model call, each one longer
    lengths = [len(m) for m in seen]
    assert all(a < b for a, b in itertools.pairwise(lengths))
    assert "[key]" in json.dumps(seen[2])  # the redacted result is what it read


def test_the_loop_imports_none_of_the_driver() -> None:
    tree = ast.parse(Path(__file__).read_text("utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported |= {f"{node.module}.{alias.name}" for alias in node.names}
    assert imported & FORBIDDEN == set()
    assert any(name.startswith("loomcore.") for name in imported)  # the check looked


def test_importing_the_components_loads_no_driver_module() -> None:
    code = (
        "import sys, loomcore.tools, loomcore.guards, loomcore.sandbox, "
        "loomcore.workspace, loomcore.window, loomcore.replay, loomcore.compose, "
        "loomcore.conversation\n"
        f"print(sorted(set(sys.modules) & {FORBIDDEN!r}))"
    )
    run = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )
    assert run.stdout.strip() == "[]"
