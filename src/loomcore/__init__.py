"""loomcore: a small agent harness.

The names a program needs to register tools and run a task. Everything else is in
its module (`loomcore.guards`, `loomcore.observability`, ...). `loomcore.anthropic`
is not here: it needs the `anthropic` extra.

Each name is loaded when it is first used, not when the package is imported, so
importing one component (`loomcore.tools`, say) never loads the driver: a hand-written
loop depends on that (spec section 1, bring your own loop).
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loomcore import finish
    from loomcore.compose import ToolKit, tool_kit
    from loomcore.eventlog import read_events
    from loomcore.limits import Limits
    from loomcore.loop import Driver, Settings
    from loomcore.ports import Config, ModelRequest, Ports, RunResult, Task
    from loomcore.prompt import DATA_RULE, PromptParts
    from loomcore.prompt import build as build_prompt
    from loomcore.replay import replay as replay_events
    from loomcore.runtime import estimate_request, estimate_tokens, system_ports

# name -> (module, attribute); no attribute means the module itself.
_EXPORTS: dict[str, tuple[str, str | None]] = {
    "DATA_RULE": ("loomcore.prompt", "DATA_RULE"),
    "Config": ("loomcore.ports", "Config"),
    "Driver": ("loomcore.loop", "Driver"),
    "Limits": ("loomcore.limits", "Limits"),
    "ModelRequest": ("loomcore.ports", "ModelRequest"),
    "Ports": ("loomcore.ports", "Ports"),
    "PromptParts": ("loomcore.prompt", "PromptParts"),
    "RunResult": ("loomcore.ports", "RunResult"),
    "Settings": ("loomcore.loop", "Settings"),
    "Task": ("loomcore.ports", "Task"),
    "ToolKit": ("loomcore.compose", "ToolKit"),
    "build_prompt": ("loomcore.prompt", "build"),
    "estimate_request": ("loomcore.runtime", "estimate_request"),
    "estimate_tokens": ("loomcore.runtime", "estimate_tokens"),
    "finish": ("loomcore.finish", None),
    "read_events": ("loomcore.eventlog", "read_events"),
    "replay_events": ("loomcore.replay", "replay"),
    "system_ports": ("loomcore.runtime", "system_ports"),
    "tool_kit": ("loomcore.compose", "tool_kit"),
}

__all__ = [
    "DATA_RULE",
    "Config",
    "Driver",
    "Limits",
    "ModelRequest",
    "Ports",
    "PromptParts",
    "RunResult",
    "Settings",
    "Task",
    "ToolKit",
    "build_prompt",
    "estimate_request",
    "estimate_tokens",
    "finish",
    "read_events",
    "replay_events",
    "system_ports",
    "tool_kit",
]


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'loomcore' has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    loaded = import_module(module)
    return loaded if attribute is None else getattr(loaded, attribute)
