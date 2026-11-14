"""The tool registry: typed Python functions become tools (spec section 3.2).

The type hints are the only source of the JSON Schema, and the docstring is the
description. A bad call comes back as `ok=False` with a message that says how to
fix it: invalid input, an unknown tool, a `ToolError`, a refusal, an `OSError`.
Nothing else is caught. Read-only tools may opt into a cache; a call that runs past
the timeout is abandoned. The registry emits no events; what the call caused
(refusals, notes) is collected by the driver's `Notices` and written by the driver.
"""

import inspect
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass, replace
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from pydantic import create_model as pydantic_model

from loomcore import finish
from loomcore.conversation import call_hash
from loomcore.events import JsonObject
from loomcore.ports import ToolResult
from loomcore.sandbox import CommandRefusedError
from loomcore.workspace import PathRefusedError, Workspace

# Not finite is refused: the log would write null where the model saw NaN.
_JSON: TypeAdapter[JsonValue] = TypeAdapter(
    JsonValue, config=ConfigDict(allow_inf_nan=False)
)
_VALUES = frozenset({"default", "const", "enum", "examples"})  # data, not schema


DEFAULT_TIMEOUT = 60.0  # seconds per call
DEFAULT_CACHE_SIZE = 128  # results kept, least recently used dropped first


class ToolError(Exception):
    """What a tool raises when the call cannot be done and the model can fix it
    (a name that does not exist, a value out of range). The message goes to the
    model and into the log, so it must not name a host path. Any other exception
    is a bug in the tool and is not caught: the run stops and is resumed from its
    log."""


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    input_schema: JsonObject

    def to_json(self) -> JsonObject:
        """The form a model request carries (the same shape as `finish.SCHEMA`)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True)
class _Tool:
    fn: Callable[..., object]
    schema: ToolSchema
    arguments: type[BaseModel]
    workspace_param: str | None
    read_only: bool
    cacheable: bool
    reads: tuple[str, ...]
    writes: tuple[str, ...]


class ToolRegistry:
    """Tools by name, built with the `Workspace` they may use."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        timeout: float | None = DEFAULT_TIMEOUT,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        self._workspace = workspace
        self._timeout = timeout
        self._cache_size = cache_size
        self._cache: OrderedDict[tuple[str, int], ToolResult] = OrderedDict()
        # Abandoned calls that may still write; the cache is bypassed while any run.
        self._abandoned = 0
        self._counting = threading.Lock()
        self._tools: dict[str, _Tool] = {}

    def register(
        self,
        fn: Callable[..., object],
        *,
        read_only: bool = False,
        cacheable: bool = False,
        reads: tuple[str, ...] = (),
        writes: tuple[str, ...] = (),
    ) -> None:
        if cacheable and not read_only:
            raise ValueError(
                f"{fn.__name__}: cacheable needs read_only=True, because a tool that "
                "writes must run every time"
            )
        tool = _build(fn, read_only, cacheable, reads, writes)
        if tool.schema.name == finish.NAME:
            raise ValueError(
                f"{finish.NAME!r} is reserved for the finish call; rename the tool"
            )
        if tool.schema.name in self._tools:
            raise ValueError(f"a tool named {tool.schema.name!r} is already registered")
        self._tools[tool.schema.name] = tool

    def schemas(self) -> list[ToolSchema]:
        return [tool.schema for tool in self._tools.values()]

    def read_only(self, name: str) -> bool:
        """For `Ports.read_only`: an unknown tool is not read-only."""
        tool = self._tools.get(name)
        return tool is not None and tool.read_only

    def execute(self, name: str, input: JsonObject) -> ToolResult:  # noqa: A002
        tool = self._tools.get(name)
        if tool is None:
            known = ", ".join(sorted(self._tools)) or "none"
            return ToolResult(False, f"Unknown tool {name!r}. The tools are: {known}.")
        key = (call_hash(name, input), self._workspace.generation)
        use_cache = tool.cacheable and self._abandoned == 0
        if use_cache and (hit := self._cache.get(key)) is not None:
            self._cache.move_to_end(key)
            return replace(hit, cached=True)
        result = self._timed(name, tool, input)
        if not tool.read_only:
            self._workspace.bump()  # a read after this write is never served stale
        if use_cache and result.ok and self._abandoned == 0:
            self._cache[key] = result
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result

    def _timed(self, name: str, tool: _Tool, input: JsonObject) -> ToolResult:  # noqa: A002
        """Run the call, abandoning it if it outlasts the timeout. Python cannot kill
        a thread, so an abandoned call may keep running: its write, if it is not
        read-only, bumps the generation again when it finally returns."""
        if self._timeout is None:
            return self._run(name, tool, input)
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(copy_context().run, self._run, name, tool, input)
        try:
            _, late = wait([future], timeout=self._timeout)
        finally:
            pool.shutdown(wait=False)
        if not late:
            return future.result()  # a bug in the tool is raised here, not hidden
        if not tool.read_only:  # `execute` bumps now; `_returned` bumps for its return
            with self._counting:
                self._abandoned += 1
            future.add_done_callback(self._returned)
        return ToolResult(
            False,
            f"{name} timed out after {self._timeout:g} s and was abandoned; it may "
            "still be running. Try a smaller input or another approach.",
        )

    def _returned(self, _: Future[ToolResult]) -> None:
        """An abandoned call that may write has finally ended: its write, if the
        workspace could not see it, is visible now, so the cache may be used again."""
        with self._counting:
            self._abandoned -= 1
        self._workspace.bump()

    def _run(self, name: str, tool: _Tool, input: JsonObject) -> ToolResult:  # noqa: A002
        try:
            parsed = tool.arguments.model_validate(input)
        except ValidationError as error:
            return ToolResult(
                False,
                f"Invalid input for {name}: {_problems(error)}. Check the arguments "
                "against the tool's schema and call it again.",
            )
        kwargs: dict[str, object] = {
            field: getattr(parsed, field) for field in tool.arguments.model_fields
        }
        if tool.workspace_param is not None:
            kwargs[tool.workspace_param] = self._workspace
        return _invoke(name, tool, kwargs)


_FIXABLE = (ToolError, PathRefusedError, CommandRefusedError, UnicodeError, OSError)


def _invoke(name: str, tool: _Tool, kwargs: dict[str, object]) -> ToolResult:
    try:
        value = tool.fn(**kwargs)
    except _FIXABLE as error:
        return ToolResult(False, f"{name} failed: {_fixable(error)}")
    try:
        return ToolResult(True, _JSON.validate_python(value))
    except ValidationError:
        return ToolResult(
            False,
            f"{name} returned something that is not JSON or not finite.",
        )


def _fixable(error: Exception) -> str:
    """What the model is told. An `OSError`'s text names the absolute host path, so
    only its `strerror` is used."""
    if isinstance(error, UnicodeError):  # a binary file, or text that cannot be written
        return "the content is not valid UTF-8 text"
    if isinstance(error, OSError):
        return (
            error.strerror or type(error).__name__
        )  # no errno: the class is all we know
    return str(error)  # written for the model: what was asked and why not


def _build(
    fn: Callable[..., object],
    read_only: bool,
    cacheable: bool,
    reads: tuple[str, ...],
    writes: tuple[str, ...],
) -> _Tool:
    doc = inspect.getdoc(fn)
    if not doc:
        raise ValueError(f"{fn.__name__} needs a docstring: it is the description")
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}
    workspace_param: str | None = None
    for param in inspect.signature(fn).parameters.values():
        if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            raise TypeError(f"{fn.__name__}: {param.name} must be a named parameter")
        if param.name not in hints:
            raise TypeError(f"{fn.__name__}: {param.name} needs a type hint")
        if hints[param.name] is Workspace:
            workspace_param = param.name  # handed in by the registry, not the model
            continue
        default = ... if param.default is param.empty else param.default
        fields[param.name] = (hints[param.name], default)
    arguments = pydantic_model(
        fn.__name__, __config__=ConfigDict(extra="forbid"), **fields
    )
    schema = ToolSchema(fn.__name__, doc, _clean(arguments.model_json_schema()))
    return _Tool(
        fn, schema, arguments, workspace_param, read_only, cacheable, reads, writes
    )


def _clean(node: Any) -> Any:  # noqa: ANN401 - walks arbitrary JSON
    """The schema without pydantic's generated titles, which only add tokens."""
    if isinstance(node, list):
        return [_clean(item) for item in node]
    if not isinstance(node, dict):
        return node
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key == "title" and isinstance(value, str):
            continue
        if key in ("properties", "$defs"):
            cleaned[key] = {name: _clean(sub) for name, sub in value.items()}
        else:
            cleaned[key] = value if key in _VALUES else _clean(value)
    return cleaned


def _problems(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in e['loc']) or 'input'}: {e['msg']}"
        for e in error.errors()
    )
