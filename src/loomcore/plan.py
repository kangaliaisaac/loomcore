"""The plan tool: the model keeps its plan in a file, and the latest plan stays in view.

`update_plan` writes `.loomcore/plan.md` through the workspace, so a human or another
tool can read it. The plan text is also the tool call's input, which is in the log; the
context store pins the exchange of the latest successful call (see
`conversation.exchanges`), so the model always sees its latest plan, with nothing read
from outside the log.
"""

from loomcore.tools import ToolError, ToolRegistry
from loomcore.workspace import Workspace

PLAN_FILE = ".loomcore/plan.md"  # not plan.md: a repository may have its own
MAX_PLAN_CHARS = 8000  # a plan is pinned, so it must not be able to fill the window


def update_plan(workspace: Workspace, text: str) -> str:
    """Replace your plan with `text`. Write the whole plan: the steps you mean to take
    and where you are in them. Your latest plan is always kept in your context."""
    plan = text.strip()
    if not plan:
        raise ToolError("the plan is empty; write the steps you mean to take")
    if len(plan) > MAX_PLAN_CHARS:
        raise ToolError(
            f"the plan is {len(plan)} characters; keep it under {MAX_PLAN_CHARS} "
            "and say it shorter"
        )
    workspace.write(PLAN_FILE, plan + "\n")
    return f"plan saved ({len(plan)} characters)"


def register(registry: ToolRegistry) -> None:
    """Add the plan tool. Its schema is in `registry.schemas()`, for `Config.tools`."""
    registry.register(update_plan, writes=(PLAN_FILE,))
