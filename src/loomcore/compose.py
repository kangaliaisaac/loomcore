"""Wiring the tool side of the ports from one `Notices` (spec 3.2).

The workspace and the sandbox report refusals and notes through callbacks fixed at
construction, and the driver collects them with the `Notices` in `Ports.notices`. If
those are two different `Notices`, a refusal inside a tool call has no bucket and is
only warned about. `tool_kit` builds the workspace and the registry on one `Notices`
and hands back the four ports that belong together, so they cannot be mixed up.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from loomcore.events import JsonObject
from loomcore.notices import Notices
from loomcore.ports import ToolResult
from loomcore.schedule import Access
from loomcore.tools import DEFAULT_CACHE_SIZE, DEFAULT_TIMEOUT, ToolRegistry
from loomcore.workspace import FilesystemWorkspace


class ToolPorts(TypedDict):
    """The fields of `Ports` that come from the tool side: splat it into `Ports`."""

    tool: Callable[[str, JsonObject], ToolResult]
    read_only: Callable[[str], bool]
    access: Callable[[str], Access]
    notices: Notices


@dataclass(frozen=True)
class ToolKit:
    notices: Notices
    workspace: FilesystemWorkspace
    registry: ToolRegistry

    def tool_ports(self) -> ToolPorts:
        """`Ports(..., **kit.tool_ports(), ...)`: the registry's three ports and the
        `Notices` the workspace reports to, which are one set by construction. Pass
        `kit.notices.refused` to a `DockerSandbox` as its `on_refusal` too."""
        return ToolPorts(
            tool=self.registry.execute,
            read_only=self.registry.read_only,
            access=self.registry.access,
            notices=self.notices,
        )


def tool_kit(
    root: Path,
    *,
    notes: str = "notes.md",
    timeout: float = DEFAULT_TIMEOUT,
    cache_size: int = DEFAULT_CACHE_SIZE,
) -> ToolKit:
    """A workspace on `root` and a registry on it, both reporting to one `Notices`."""
    notices = Notices()
    workspace = FilesystemWorkspace(
        root, notes=notes, on_refusal=notices.refused, on_note=notices.noted
    )
    registry = ToolRegistry(workspace, timeout=timeout, cache_size=cache_size)
    return ToolKit(notices, workspace, registry)
