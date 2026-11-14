"""The driver's ports and value types (spec section 1).

A port is the only place the driver touches the world: the model, the tools, the
clocks, the sleep and the price list. Tests pass scripted fakes.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal, Protocol

from pydantic import JsonValue

from loomcore.events import JsonObject
from loomcore.limits import Limits
from loomcore.notices import Notices
from loomcore.retry import CallError, Random


@dataclass(frozen=True)
class Config:
    """What a resuming process must rebuild exactly; its digest is in the log."""

    model: str
    system: str
    tools: tuple[JsonObject, ...] = ()
    # What the switch-strategy rung changes to (spec section 1). Not part of the
    # digest: the log holds this configuration and its digest once it is in force.
    switch_to: Config | None = None

    def to_json(self) -> JsonObject:
        return {"model": self.model, "system": self.system, "tools": list(self.tools)}

    def digest(self) -> str:
        canonical = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class Task:
    prompt: JsonValue
    limits: Limits


@dataclass(frozen=True)
class ModelRequest:
    model: str
    system: str
    messages: list[JsonObject]
    tools: tuple[JsonObject, ...]
    max_tokens: int


class ModelCallError(Exception):
    """The model port raises this when a call fails; the retry policy reads it."""

    def __init__(self, error: CallError) -> None:
        super().__init__(f"model call failed: {error}")
        self.error = error


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: JsonValue
    cached: bool = False  # served from the registry's cache; the tool did not run


class Pricing(Protocol):
    """Prices from recorded tokens; the vendored table arrives with card 10."""

    def worst_case(self, request: ModelRequest) -> Decimal:
        """The request's known input plus `max_tokens` of output."""
        ...

    def cost(self, message: JsonObject) -> tuple[Decimal, str]:
        """The call's cost and the `pricing_version` it was priced with."""
        ...


@dataclass(frozen=True)
class Ports:
    model: Callable[[ModelRequest], JsonObject]  # raises ModelCallError
    # ok=false for a failure the model can fix; any other exception is a bug in the
    # tool and stops the run. The tool's timeout is the registry's (spec 3.2).
    tool: Callable[[str, JsonObject], ToolResult]
    read_only: Callable[[str], bool]  # by tool name
    notices: Notices  # the callbacks of the workspace and sandbox report here
    pricing: Pricing
    clock: Callable[[], datetime]  # wall-clock UTC
    monotonic: Callable[[], float]  # durations
    random: Random  # jitter
    sleep: Callable[[float], None]
    new_run_id: Callable[[], str]


@dataclass(frozen=True)
class RunRef:
    session_id: str
    run_id: str


@dataclass(frozen=True)
class Decision:
    granted: bool
    by: str
    note: str = ""


@dataclass(frozen=True)
class RunResult:
    ref: RunRef
    status: Literal["finished", "paused"]
    reason: str  # the run_finished reason, or the run_paused reason
    limit: str | None = None  # which limit, when paused on one
    question: str | None = None  # what the human is asked, when awaiting approval
    result: JsonValue = None
