"""The system-prompt builder (spec section 3.1).

A prompt is an ordered list of blocks: role, rules and tool docs form the stable
prefix, and the environment, the one part that may change from run to run, comes
after it. Building is pure and deterministic, so the same parts always give the same
bytes, and a changed part can only change the blocks from itself onwards.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from loomcore.tools import ToolSchema

BlockName = Literal["role", "rules", "tools", "environment"]
STABLE: frozenset[BlockName] = frozenset(
    {"role", "rules", "tools"}
)  # before the breakpoint
SEPARATOR = "\n\n"


@dataclass(frozen=True)
class PromptParts:
    role: str
    rules: tuple[str, ...] = ()
    tools: tuple[ToolSchema, ...] = ()
    environment: str = ""  # the only part allowed to vary; it follows the breakpoint


@dataclass(frozen=True)
class Block:
    name: BlockName
    text: str


@dataclass(frozen=True)
class SystemPrompt:
    blocks: tuple[Block, ...]

    @property
    def text(self) -> str:
        """The whole prompt, for `Config.system`."""
        return SEPARATOR.join(block.text for block in self.blocks)

    @property
    def cache_breakpoint(self) -> int | None:
        """The index of the last stable block, where the model port puts its cache
        marker; None when there is no stable block."""
        stable = [k for k, block in enumerate(self.blocks) if block.name in STABLE]
        return stable[-1] if stable else None

    @property
    def stable_text(self) -> str:
        """The prefix up to and including the cache breakpoint: what goes in
        `Config.system`, so the whole `system` string of a request is the part the
        model port caches."""
        end = self.cache_breakpoint
        kept = () if end is None else self.blocks[: end + 1]
        return SEPARATOR.join(block.text for block in kept)

    @property
    def variable_text(self) -> str:
        """What follows the breakpoint (the environment): added by the model port
        after the cached prefix, outside the configuration digest."""
        end = self.cache_breakpoint
        rest = self.blocks if end is None else self.blocks[end + 1 :]
        return SEPARATOR.join(block.text for block in rest)


def build(parts: PromptParts) -> SystemPrompt:
    """Assemble the parts in their fixed order. An empty part leaves out its block.

    Raises `ValueError` for two tools with one name: their order would decide the
    bytes, and a model cannot tell them apart either.
    """
    names = [tool.name for tool in parts.tools]
    if len(set(names)) != len(names):
        raise ValueError("two tools share a name; a prompt lists each tool once")
    candidates = (
        Block("role", parts.role.strip()),
        Block("rules", _list("Rules:", (_line(rule) for rule in parts.rules))),
        Block("tools", _tool_docs(parts.tools)),
        Block("environment", _section("Environment:", parts.environment)),
    )
    return SystemPrompt(tuple(block for block in candidates if block.text))


def _tool_docs(tools: Iterable[ToolSchema]) -> str:
    # By name, so the order tools were registered in cannot change the bytes.
    docs = (f"{_line(t.name)}: {_line(t.description)}" for t in _by_name(tools))
    return _list("Tools:", docs)


def _line(text: str) -> str:
    """One line: no newline can fake a heading or a block boundary."""
    return " ".join(text.split())


def _by_name(tools: Iterable[ToolSchema]) -> list[ToolSchema]:
    return sorted(tools, key=lambda tool: tool.name)


def _list(heading: str, lines: Iterable[str]) -> str:
    kept = [line for line in lines if line]
    return heading + "\n" + "\n".join(f"- {line}" for line in kept) if kept else ""


def _section(heading: str, body: str) -> str:
    stripped = body.strip()
    return f"{heading}\n{stripped}" if stripped else ""
