"""Scripted ports for driving a run: a fake model, fake tools, a clock, a sleep."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from factories import START
from pydantic import JsonValue

from loomcore.events import Event, JsonObject
from loomcore.limits import Limits
from loomcore.loop import Driver
from loomcore.notices import Notices
from loomcore.ports import (
    Config,
    ModelCallError,
    ModelRequest,
    Ports,
    RunRef,
    ToolResult,
)
from loomcore.retry import CallError
from loomcore.sandbox import Refusal
from loomcore.schedule import Access

CONFIG = Config(model="m", system="be brief", tools=({"name": "search"},))
CAPS = Limits(run_cap=None, month_cap=None)
COST = Decimal("0.01")


def fake_count(messages: Sequence[JsonObject]) -> int:
    """A deterministic stand-in for the edge's counter (the serialised length / 4)."""
    return len(json.dumps(messages, sort_keys=True)) // 4 + 1


class ToolBugError(Exception):
    """Raised by a fake tool to stand for a bug nobody planned for."""


class Crash(BaseException):
    """Raised by a fake port to stand for the process dying mid-run."""


def reply(text: str = "done", stop: str = "end_turn") -> JsonObject:
    return {"content": [{"type": "text", "text": text}], "stop_reason": stop}


def use(
    *calls: tuple[str, str], stop: str = "tool_use", q: str | None = None
) -> JsonObject:
    """A response that calls tools: pairs of (tool_use_id, tool name).

    The input is `{"q": id}` unless `q` is given, so calls with the same `q` repeat.
    """
    blocks: list[JsonValue] = [
        {"type": "tool_use", "id": i, "name": name, "input": {"q": q or i}}
        for i, name in calls
    ]
    return {"content": blocks, "stop_reason": stop}


def finish(result: JsonValue, *others: tuple[str, str]) -> JsonObject:
    """A response that calls the finish tool, beside any other tool calls."""
    blocks: list[JsonValue] = [
        {"type": "tool_use", "id": "fin", "name": "finish", "input": {"result": result}}
    ]
    blocks += [
        {"type": "tool_use", "id": i, "name": name, "input": {"q": i}}
        for i, name in others
    ]
    return {"content": blocks, "stop_reason": "tool_use"}


def failure(status: int, retry_after: float | None = None) -> ModelCallError:
    return ModelCallError(CallError("http", status, retry_after))


@dataclass
class Clock:
    now: datetime = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakePricing:
    fail = False  # a model the price table does not list

    def worst_case(self, request: ModelRequest) -> Decimal:
        return COST

    def cost(self, message: JsonObject) -> tuple[Decimal, str]:
        if self.fail:
            raise LookupError("no price for this model")
        return COST, "test"


@dataclass
class World:
    """Everything a test arranges and inspects."""

    script: list[JsonObject | Exception] = field(default_factory=list)
    requests: list[ModelRequest] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    tool_results: dict[str, ToolResult] = field(default_factory=dict)
    crash_in: set[str] = field(default_factory=set)  # tool_use ids that kill the run
    bug_in: set[str] = field(default_factory=set)  # tool_use ids whose tool breaks
    noted_in: dict[str, str] = field(default_factory=dict)  # id -> notes path written
    refused_in: dict[str, Refusal] = field(default_factory=dict)  # id -> refusal
    notices: Notices = field(default_factory=Notices)
    together: set[str] = field(default_factory=set)  # tools that may run together
    declared: dict[str, Access] = field(default_factory=dict)  # reads and writes
    read_only: set[str] = field(default_factory=set)  # tool names
    sleeps: list[float] = field(default_factory=list)
    tool_seconds: float = 0  # how long each tool call takes on the fake clock
    crash_on_call: int | None = None  # the Nth tool call kills the run
    crash_on_model: int | None = None  # the Nth model call kills the run
    calls_made: int = 0
    pricing: FakePricing = field(default_factory=FakePricing)
    clock: Clock = field(default_factory=Clock)
    ids: int = 0

    def model(self, request: ModelRequest) -> JsonObject:
        self.requests.append(request)
        if len(self.requests) == self.crash_on_model:
            raise Crash
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def tool(self, name: str, args: JsonObject) -> ToolResult:
        use_id = str(args["q"])
        self.calls_made += 1
        if self.calls_made == self.crash_on_call:
            raise Crash
        self.clock.advance(self.tool_seconds)
        if use_id in self.crash_in:
            self.crash_in.discard(use_id)
            raise Crash
        if use_id in self.refused_in:
            self.notices.refused(self.refused_in[use_id])
        if use_id in self.noted_in:
            self.notices.noted(self.noted_in[use_id])
        if use_id in self.bug_in:
            raise ToolBugError(use_id)
        self.tool_calls.append(use_id)
        return self.tool_results.get(use_id, ToolResult(True, f"result {use_id}"))

    def access(self, name: str) -> Access:
        """Every tool conflicts with every other unless the test names it `together`
        (touching nothing) or gives its `declared` reads and writes."""
        if name in self.declared:
            return self.declared[name]
        return Access(read_only=name in self.together)

    def new_run_id(self) -> str:
        self.ids += 1
        return f"r{self.ids}"

    def ports(self) -> Ports:
        return Ports(
            model=self.model,
            tool=self.tool,
            read_only=lambda name: name in self.read_only,
            access=self.access,
            notices=self.notices,
            pricing=self.pricing,
            count_tokens=fake_count,
            clock=self.clock,
            monotonic=lambda: 0.0,
            random=lambda: 0.5,
            sleep=self.sleeps.append,
            new_run_id=self.new_run_id,
        )

    def driver(self, tmp: Path, config: Config = CONFIG) -> Driver:
        return Driver(config, self.ports(), tmp / "s1.jsonl", tmp / "spend.jsonl")


def ref(run_id: str = "r1") -> RunRef:
    return RunRef("s1", run_id)


def kinds(events: tuple[Event, ...], run_id: str = "r1") -> list[str]:
    return [e.kind for e in events if e.run_id == run_id]
