"""Guards: checks on a tool call before it runs and on its result before it is logged
(spec 3.7).

Everything here is pure. A pre guard says `Allow`, `Deny` or `Ask` about a request; a
post guard says `Allow`, `Deny` or `Redact` about a result. `run_pre` and `run_post`
hold the ordering rules, and a driver or a hand-written loop (f24) calls them; they
write nothing, so the caller logs what they return.
"""

from dataclasses import dataclass, replace
from typing import Protocol

from loomcore.events import JsonObject
from loomcore.ports import ToolResult
from loomcore.replay import RunState


@dataclass(frozen=True)
class ToolRequest:
    """The call a guard judges. A guard must not change `input`: it is the object
    the call runs with and the log holds."""

    tool_use_id: str
    name: str
    input: JsonObject
    call_hash: str


@dataclass(frozen=True)
class Allow:
    pass


@dataclass(frozen=True)
class Deny:
    # logged, and the observation the model sees (instead of the call or its result)
    reason: str


@dataclass(frozen=True)
class Ask:
    question: str  # a pre guard only: the human is asked before the tool runs


@dataclass(frozen=True)
class Redact:
    content: str  # replaces the result's content, in the log and for the model
    reason: str  # what was removed and why, with no secret in it


PreDecision = Allow | Deny | Ask
PostDecision = Allow | Deny | Redact


class GuardError(Exception):
    """A guard returned a decision its stage does not allow (`Redact` from a pre
    guard, `Ask` from a post guard): the guard is wrong, not the run."""


class Guard(Protocol):
    """Stage `pre`: before the tool runs."""

    def check(self, request: ToolRequest, state: RunState) -> PreDecision: ...


class PostGuard(Protocol):
    """Stage `post`: on the result, before it is logged."""

    def check(
        self, request: ToolRequest, result: ToolResult, state: RunState
    ) -> PostDecision: ...


@dataclass(frozen=True)
class Guards:
    pre: tuple[Guard, ...] = ()
    post: tuple[PostGuard, ...] = ()


@dataclass(frozen=True)
class Screened:
    """A result after the post guards: what to complete the call with, and each
    decision that was not an `Allow`, in the order the guards made them."""

    result: ToolResult
    decisions: tuple[Deny | Redact, ...] = ()


def run_pre(guards: Guards, request: ToolRequest, state: RunState) -> PreDecision:
    """The first `Deny` in guard order if any guard denies (later guards are then not
    asked), else the first `Ask`, else `Allow`. A guard that asks does not stop the
    others: a human must never approve a call that a later guard would refuse."""
    asking: Ask | None = None
    for guard in guards.pre:
        decision = guard.check(request, state)
        match decision:
            case Allow():
                continue
            case Deny():
                return decision
            case Ask():
                asking = asking or decision
            case _:
                raise GuardError(f"a pre guard returned {type(decision).__name__}")
    return asking or Allow()


def run_post(
    guards: Guards, request: ToolRequest, result: ToolResult, state: RunState
) -> Screened:
    """Each guard sees the result the one before left. A `Redact` replaces the
    content (`ok` and the rest stay) and the pipeline goes on; the first `Deny`
    replaces the result with an `ok=False` observation of its reason and stops."""
    current = result
    decisions: list[Deny | Redact] = []
    for guard in guards.post:
        decision = guard.check(request, current, state)
        match decision:
            case Allow():
                continue
            case Redact():
                current = replace(current, content=decision.content)
                decisions.append(decision)
            case Deny():
                decisions.append(decision)
                denied = ToolResult(
                    False, decision.reason, cached=current.cached, from_tool=False
                )
                return Screened(denied, tuple(decisions))
            case _:
                raise GuardError(f"a post guard returned {type(decision).__name__}")
    return Screened(current, tuple(decisions))
