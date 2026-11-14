"""f19, part 1: the guard pipelines are pure and ordered."""

import pytest

from loomcore.guards import (
    Allow,
    Ask,
    Deny,
    GuardError,
    Guards,
    PostDecision,
    PreDecision,
    Redact,
    ToolRequest,
    run_post,
    run_pre,
)
from loomcore.ports import ToolResult
from loomcore.replay import RunState

REQUEST = ToolRequest("t1", "search", {"q": "x"}, "hash")
STATE = RunState()


class Says:
    """A pre guard that answers one thing and notes it was asked."""

    def __init__(self, decision: PreDecision) -> None:
        self.decision = decision
        self.asked = 0

    def check(self, request: ToolRequest, state: RunState) -> PreDecision:
        self.asked += 1
        return self.decision


class Rewrites:
    """A post guard that answers one thing and keeps what it was shown."""

    def __init__(self, decision: PostDecision) -> None:
        self.decision = decision
        self.saw: list[ToolResult] = []

    def check(
        self, request: ToolRequest, result: ToolResult, state: RunState
    ) -> PostDecision:
        self.saw.append(result)
        return self.decision


def test_with_no_pre_guard_or_only_allows_a_call_is_allowed() -> None:
    assert run_pre(Guards(), REQUEST, STATE) == Allow()
    assert (
        run_pre(Guards(pre=(Says(Allow()), Says(Allow()))), REQUEST, STATE) == Allow()
    )


def test_the_first_deny_or_ask_in_order_wins_and_later_guards_are_not_asked() -> None:
    later = Says(Deny("never"))
    deny_first = Guards(pre=(Says(Allow()), Says(Deny("no")), later))
    assert run_pre(deny_first, REQUEST, STATE) == Deny("no")
    assert later.asked == 0
    ask_first = Guards(pre=(Says(Ask("sure?")), Says(Deny("no"))))
    assert run_pre(ask_first, REQUEST, STATE) == Ask("sure?")
    deny_then_ask = Guards(pre=(Says(Deny("no")), Says(Ask("sure?"))))
    assert run_pre(deny_then_ask, REQUEST, STATE) == Deny("no")


def test_a_pre_guard_cannot_redact() -> None:
    with pytest.raises(GuardError, match="Redact"):
        run_pre(Guards(pre=(Says(Redact("x", "why")),)), REQUEST, STATE)  # type: ignore[arg-type]


def test_a_result_no_post_guard_objects_to_comes_back_untouched() -> None:
    result = ToolResult(True, {"n": 1}, cached=True)
    screened = run_post(Guards(post=(Rewrites(Allow()),)), REQUEST, result, STATE)
    assert screened.result is result
    assert screened.decisions == ()
    assert run_post(Guards(), REQUEST, result, STATE).result is result


def test_a_redaction_replaces_the_content_and_keeps_the_rest() -> None:
    result = ToolResult(False, "key=SECRET", cached=True)
    guards = Guards(post=(Rewrites(Redact("key=[removed]", "an api key")),))
    screened = run_post(guards, REQUEST, result, STATE)
    assert screened.result == ToolResult(False, "key=[removed]", cached=True)
    assert screened.decisions == (Redact("key=[removed]", "an api key"),)


def test_each_post_guard_sees_what_the_one_before_left() -> None:
    first = Rewrites(Redact("step one", "first"))
    second = Rewrites(Redact("step two", "second"))
    third = Rewrites(Allow())
    screened = run_post(
        Guards(post=(first, second, third)), REQUEST, ToolResult(True, "raw"), STATE
    )
    assert [r.content for r in first.saw] == ["raw"]
    assert [r.content for r in second.saw] == ["step one"]
    assert [r.content for r in third.saw] == ["step two"]
    assert screened.result.content == "step two"
    assert [d.reason for d in screened.decisions if isinstance(d, Redact)] == [
        "first",
        "second",
    ]


def test_a_post_deny_replaces_the_result_with_its_reason_and_stops() -> None:
    after = Rewrites(Redact("never", "x"))
    guards = Guards(
        post=(Rewrites(Redact("clean", "r")), Rewrites(Deny("too big")), after)
    )
    screened = run_post(guards, REQUEST, ToolResult(True, "raw", cached=True), STATE)
    assert screened.result == ToolResult(False, "too big")  # not ok, not cached
    assert screened.decisions == (Redact("clean", "r"), Deny("too big"))
    assert after.saw == []


def test_a_post_guard_cannot_ask_because_the_tool_has_already_run() -> None:
    with pytest.raises(GuardError, match="Ask"):
        run_post(
            Guards(post=(Rewrites(Ask("sure?")),)),  # type: ignore[arg-type]
            REQUEST,
            ToolResult(True, "x"),
            STATE,
        )


def test_the_decisions_are_values() -> None:
    assert Deny("a") == Deny("a")
    assert Ask("q") != Ask("other")
    assert hash(Redact("c", "r")) == hash(Redact("c", "r"))
