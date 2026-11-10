"""The sandbox boundary (spec section 3.3): what a refusal says.

The rule evaluation and the process runner join this module with f12; the workspace
(workspace.py) already reports its refusals in this shape.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Refusal:
    """What `sandbox_denied` logs: the action tried, its target, and why not."""

    action: str
    target: str
    reason: str
