"""The scripted three-step eval (spec section 5).

A task in the shape of the capstones: read a small filing excerpt, compute one
figure, write an answer that cites its source. Everything is real except the ports that
stand for the outside world (the model, both clocks, the token counter, the jitter,
the sleep and the run ids): the driver, the tool registry on a filesystem
workspace, the system-prompt builder, the vendored price table and the event log. It
imports nothing from `tests/`, so it can be read, and run, on its own.

The excerpt is invented for this eval; it is not a real company's filing.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from pydantic import JsonValue

from loomcore import finish
from loomcore.compose import ToolKit, tool_kit
from loomcore.events import JsonObject
from loomcore.limits import Limits
from loomcore.loop import Driver, Settings
from loomcore.observability.cost import TablePricing, vendored_table
from loomcore.ports import Config, ModelRequest, Ports, RunResult, Task
from loomcore.prompt import DATA_RULE, PromptParts, build
from loomcore.workspace import Workspace

MODEL = "claude-sonnet-5-5"

FILING = (
    "Example Corp, annual report excerpt (invented for this eval)\n"
    "Fiscal year 2025, in millions of dollars\n"
    "Revenue: 2,000\n"
    "Net income: 250\n"
)

TASK = (
    "Read filing.txt. Work out the net margin for fiscal 2025 (net income divided "
    "by revenue), write the figure and the line it came from to answer.md, then "
    "finish with the figure."
)

ANSWER = (
    "Net margin, fiscal 2025: 12.5% "
    "(net income 250 / revenue 2,000; filing.txt, lines 3 and 4)\n"
)

# The tokens each canned response reports, so the run's price can be worked out by
# hand: (input, output, cache read), in the order the model is called.
USAGE: tuple[tuple[int, int, int], ...] = (
    (1000, 100, 0),
    (1500, 150, 1000),
    (2000, 80, 2500),
)


def read_file(workspace: Workspace, path: str) -> str:
    """Read a text file in the workspace."""
    return workspace.read(path)


def write_file(workspace: Workspace, path: str, content: str) -> str:
    """Write a text file in the workspace, replacing what was there."""
    workspace.write(path, content)
    return f"wrote {len(content)} characters to {path}"


def _response(n: int, blocks: list[JsonObject], stop: str = "tool_use") -> JsonObject:
    """A response shaped like the Messages API's, with the usage of call `n`."""
    fresh, output, cached = USAGE[n]
    return {
        "id": f"msg_{n + 1}",
        "model": MODEL,
        "role": "assistant",
        "content": list(blocks),
        "stop_reason": stop,
        "usage": {
            "input_tokens": fresh,
            "output_tokens": output,
            "cache_read_input_tokens": cached,
        },
    }


def _call(use_id: str, name: str, **args: JsonValue) -> JsonObject:
    return {"type": "tool_use", "id": use_id, "name": name, "input": dict(args)}


def canned_responses() -> list[JsonObject]:
    """What the model says at each of the three steps."""
    return [
        _response(0, [_call("t1", "read_file", path="filing.txt")]),
        _response(1, [_call("t2", "write_file", path="answer.md", content=ANSWER)]),
        _response(2, [_call("t3", finish.NAME, result="12.5%")]),
    ]


@dataclass
class ScriptedModel:
    """The model port: replays the canned responses in order and keeps each request."""

    script: list[JsonObject] = field(default_factory=canned_responses)
    requests: list[ModelRequest] = field(default_factory=list)

    def __call__(self, request: ModelRequest) -> JsonObject:
        self.requests.append(request)
        return self.script.pop(0)


@dataclass
class Clock:
    """A fake wall clock that moves a second each time it is read."""

    now: datetime = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def count_tokens(messages: Sequence[JsonObject]) -> int:
    """A deterministic stand-in for the edge's counter: serialised length over four."""
    return len(json.dumps(messages, sort_keys=True)) // 4 + 1


def count_request(request: ModelRequest) -> int:
    return count_tokens([{"system": request.system}, *request.messages])


@dataclass
class Eval:
    """One run of the eval and everything a check looks at afterwards."""

    session: Path
    result: RunResult
    model: ScriptedModel
    task: Task


def tools_and_config(kit: ToolKit) -> Config:
    kit.registry.register(read_file, read_only=True)
    kit.registry.register(write_file, writes=("answer.md",))
    schemas = tuple(schema.to_json() for schema in kit.registry.schemas())
    prompt = build(
        PromptParts(
            role="You answer questions about a company's filing.",
            rules=("Cite the line a figure came from.", DATA_RULE),
            tools=tuple(kit.registry.schemas()),
        )
    )
    return Config(
        model=MODEL,
        system=prompt.text,
        tools=(*schemas, finish.SCHEMA),
        cache_prefix=len(prompt.stable_text),
    )


def driver_on(root: Path, model: ScriptedModel) -> tuple[Driver, Task]:
    """A driver with its files under `root`; only the model and the clocks are faked."""
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "filing.txt").write_text(FILING, encoding="utf-8")
    kit = tool_kit(work)
    config = tools_and_config(kit)
    ticks = iter(range(1_000_000))
    ports = Ports(
        model=model,
        pricing=TablePricing(vendored_table(), count_request),
        count_tokens=count_tokens,
        clock=Clock(),
        monotonic=lambda: float(next(ticks)),
        random=lambda: 0.5,
        sleep=lambda _seconds: None,
        new_run_id=lambda: "run-1",
        **kit.tool_ports(),
    )
    session = root / "session.jsonl"
    spend = root / "spend.jsonl"
    driver = Driver(config, ports, Settings(session, spend))
    limits = Limits(run_cap=Decimal("1"), month_cap=Decimal("10"))
    return driver, Task(TASK, limits)


def run_eval(root: Path) -> Eval:
    """Run the task to its end in `root`: the three canned steps."""
    model = ScriptedModel()
    driver, task = driver_on(root, model)
    result = driver.run(task)
    return Eval(root / "session.jsonl", result, model, task)
