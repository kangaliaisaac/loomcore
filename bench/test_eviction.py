"""The eviction bench scores what it says it scores."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from eviction import (
    SEEDED_BUDGET,
    Ref,
    completed,
    log_task_tokens,
    main,
    optimal_faults,
    policy_faults,
    ref_for,
    references,
    replay_policy,
    report,
    score,
    score_log,
    seeded_workload,
)
from hypothesis import given
from hypothesis import strategies as st

from loomcore import estimate_tokens
from loomcore.conversation import exchanges, recount
from loomcore.events import Event, JsonObject, make_event
from loomcore.limits import Limits
from loomcore.window import Budget, Carried, build_window

NOW = datetime(2026, 11, 16, tzinfo=UTC)
LIMITS = Limits(run_cap=Decimal(1), month_cap=Decimal(1))


def refs(keys: str, tokens: int = 1, stub: int = 20) -> list[Ref]:
    return [Ref(key, tokens, stub) for key in keys]


def test_a_hand_worked_string() -> None:
    # Room for two pages. At c, furthest in future drops b (never asked again) and
    # keeps a, which the next request needs. The context store's rule drops the
    # oldest page, a, so the last request faults.
    string = refs("abca")
    assert optimal_faults(string, 2) == 0
    assert policy_faults(string, 2) == 1


def test_room_for_everything_means_no_faults() -> None:
    string = refs("abcabcabc")
    assert (policy_faults(string, 100), optimal_faults(string, 100)) == (0, 0)


def test_a_first_request_is_not_a_fault() -> None:
    result = score(refs("abcd"), 1)
    assert (result.compulsory, result.policy, result.furthest) == (4, 0, 0)
    assert result.avoidable == 0


def test_an_old_result_is_cleared_when_the_window_reaches_compact_at() -> None:
    # Two 400-token pages fill 800 of 1000: compaction starts at 80% and clears the
    # oldest result, so the third request, for the first page, faults although
    # nothing was dropped. Furthest in future keeps both pages and does not.
    string = refs("aba", tokens=400)
    assert policy_faults(string, 1000) == 1
    assert optimal_faults(string, 1000) == 0


def test_a_page_larger_than_the_budget_makes_an_over_window_that_records_nothing() -> (
    None
):
    # The newest page alone is over the budget: the driver sends the window as it
    # was and drops nothing, so the first page is still shown when it is asked for
    # again. Furthest in future must make room, so it evicts it.
    big = refs("aba", tokens=50)
    assert policy_faults(big, 10) == 0
    assert optimal_faults(big, 10) == 1


def test_the_task_takes_its_share_of_the_budget() -> None:
    # Two 400-token pages in 1700 stay under compact_at (1360), so both are shown and
    # the third request is a hit. A 600-token task takes the window to 1400, which
    # compaction clears, and the third request faults. Furthest in future has 1100
    # left, which holds both pages.
    string = refs("aba", tokens=400)
    assert policy_faults(string, 1700, task_tokens=0) == 0
    assert policy_faults(string, 1700, task_tokens=600) == 1
    assert optimal_faults(string, 1700, task_tokens=600) == 0
    assert optimal_faults(string, 1700, task_tokens=1000) == 1  # 700 left: one page


def test_pages_are_ranked_by_the_drivers_item_numbers_not_by_position() -> None:
    # No page can be cleared (its stub is bigger than it), so only value per token
    # decides, and value is (item number + 1) over tokens. `a` is item 1 and `b` item
    # 2 in the driver, so with room for one of them beside `c` the store keeps `a`
    # (2/100 against 3/200) and `b` faults. Numbering them by position in the string,
    # where three hits for `a` push `b` to 4, would keep `b` and give no fault.
    string = [
        Ref("a", 100, 999), Ref("a", 100, 999), Ref("a", 100, 999), Ref("a", 100, 999),
        Ref("b", 200, 999),
        Ref("c", 150, 999),
        Ref("b", 200, 999),
    ]  # fmt: skip
    assert policy_faults(string, 400) == 1
    assert optimal_faults(string, 400) == 0  # it drops `a`, which is never asked again


@given(
    st.text(alphabet="abcdefg", min_size=1, max_size=60),
    st.integers(min_value=1, max_value=7),
    st.sampled_from([1, 100]),  # smaller than a cleared stub, and larger
)
def test_with_equal_pages_furthest_in_future_is_never_beaten(
    keys: str, room: int, size: int
) -> None:
    # CLRS 15.4: for pages of one size it is the best possible, so no policy
    # (this one included) can fault less.
    string = refs(keys, size)
    assert optimal_faults(string, room * size) <= policy_faults(string, room * size)


def test_a_page_is_the_size_of_the_drivers_item_and_its_cleared_form() -> None:
    done = completed("t1", "x" * 3000)
    page = ref_for("k", "read_file", {"path": "a.py"}, done)
    use: JsonObject = {
        "type": "tool_use",
        "id": "t1",
        "name": "read_file",
        "input": {"path": "a.py"},
    }
    call = make_event(
        "model_call",
        seq=1,
        timestamp=NOW,
        session_id="s",
        run_id="r",
        message={"role": "assistant", "content": [use], "stop_reason": "tool_use"},
        duration_ms=1,
        cost_usd=Decimal(0),
        pricing_version="v",
    )
    item = recount(exchanges("task", [call, done], estimate_tokens), estimate_tokens)[1]
    assert item.cleared is not None
    assert (page.tokens, page.stub) == (item.tokens, item.cleared.tokens)
    assert 20 < page.stub < page.tokens  # a stub is far from nothing


def test_the_seeded_workload_is_the_same_every_time_and_has_a_gap() -> None:
    assert seeded_workload() == seeded_workload()
    result = score(seeded_workload(), SEEDED_BUDGET)
    assert result.compulsory == 36
    assert result.avoidable == 264
    assert (result.policy, result.furthest) == (201, 44)  # from the rules above
    assert result.furthest < result.policy


def _log() -> list[Event]:
    def at(seq: int, kind: str, **fields: object) -> Event:
        return make_event(
            kind, seq=seq, timestamp=NOW, session_id="s", run_id="r", **fields
        )

    def call(seq: int, use: str, name: str, digest: str) -> Event:
        return at(
            seq,
            "tool_requested",
            tool_use_id=use,
            name=name,
            input={},
            call_hash=digest,
        )

    def done(seq: int, use: str, content: str, ok: bool = True) -> Event:
        return at(
            seq,
            "tool_completed",
            tool_use_id=use,
            ok=ok,
            content=content,
            duration_ms=1,
            cached=False,
        )

    return [
        at(
            0,
            "run_started",
            task="Read the file and summarise it.",
            config={},
            limits=LIMITS.to_json(),
            config_digest="d",
        ),
        call(1, "t1", "read", "h1"),
        done(2, "t1", "x" * 300),
        call(3, "t2", "read", "h1"),  # the same call, the same result
        done(4, "t2", "x" * 300),
        call(5, "t3", "read", "h1"),  # the same call after a write: a new result
        done(6, "t3", "y" * 300),
        call(7, "t4", "read", "h2"),  # refused: no result to keep
        done(8, "t4", "no such file", ok=False),
        call(9, "t5", "finish", "h3"),  # ends the run: not a page
        done(10, "t5", "finished"),
        call(11, "t6", "stat", "h4"),  # never completed: no result to size
        call(12, "t7", "update_plan", "h5"),  # the plan is pinned, not a page
        done(13, "t7", "wrote the plan"),
    ]


def test_a_logs_requests_are_its_successful_calls_keyed_by_call_and_result() -> None:
    string = references(_log())
    assert [r.key.split(":")[0] for r in string] == ["h1", "h1", "h1"]
    assert string[0].key == string[1].key != string[2].key  # a new result is a new key
    assert all(r.tokens > r.stub > 0 for r in string)


def test_a_log_with_more_than_one_run_is_refused() -> None:
    fork = make_event(
        "run_forked",
        seq=99,
        timestamp=NOW,
        session_id="s",
        run_id="r2",
        **{"from": {"runId": "r", "seq": 3}},
    )
    second = make_event(
        "run_started",
        seq=100,
        timestamp=NOW,
        session_id="s",
        run_id="r3",
        task="Another task.",
        config={},
        limits=LIMITS.to_json(),
        config_digest="d",
    )
    for other in (fork, second):
        with pytest.raises(ValueError, match="more than one run"):
            references([*_log(), other])


def test_a_logs_score_uses_the_runs_own_compaction_fractions() -> None:
    def run(limits: Limits) -> list[Event]:
        def at(seq: int, kind: str, **fields: object) -> Event:
            return make_event(
                kind, seq=seq, timestamp=NOW, session_id="s", run_id="r", **fields
            )

        events = [
            at(
                0,
                "run_started",
                task="t",
                config={},
                limits=limits.to_json(),
                config_digest="d",
            )
        ]
        for n, name in enumerate("aba"):
            events += [
                at(
                    1 + 2 * n,
                    "tool_requested",
                    tool_use_id=f"t{n}",
                    name="read",
                    input={},
                    call_hash=name,
                ),
                at(
                    2 + 2 * n,
                    "tool_completed",
                    tool_use_id=f"t{n}",
                    ok=True,
                    content=name * 1100,
                    duration_ms=1,
                    cached=False,
                    from_tool=True,
                ),
            ]
        return events

    eager = Limits(
        run_cap=Decimal(1), month_cap=Decimal(1), compact_at=0.5, compact_to=0.3
    )
    lazy = Limits(
        run_cap=Decimal(1), month_cap=Decimal(1), compact_at=0.99, compact_to=0.3
    )
    # Two pages fill most of the budget: an eager run compacts and clears the first
    # page, so asking for it again faults; a lazy one never reaches its threshold.
    assert score_log(run(eager), 1100).policy == 1
    assert score_log(run(lazy), 1100).policy == 0


def test_a_resume_that_changes_the_compaction_fractions_is_refused() -> None:
    def resumed(limits: Limits) -> Event:
        return make_event(
            "run_resumed",
            seq=50,
            timestamp=NOW,
            session_id="s",
            run_id="r",
            reason="limit",
            resume_from=12,
            limits=limits.to_json(),
        )

    same = Limits(run_cap=Decimal(2), month_cap=Decimal(2))  # new caps, same fractions
    other = Limits(
        run_cap=Decimal(1), month_cap=Decimal(1), compact_at=0.5, compact_to=0.3
    )
    assert score_log([*_log(), resumed(same)], 500).requests == 3
    with pytest.raises(ValueError, match="compaction fractions"):
        score_log([*_log(), resumed(other)], 500)


def test_a_logs_score_counts_its_task_and_a_log_without_one_counts_none() -> None:
    events = _log()
    task = estimate_tokens(
        [{"role": "user", "content": "Read the file and summarise it."}]
    )
    assert log_task_tokens(events) == task > 0
    assert log_task_tokens(events[1:]) == 0
    assert score_log(events, 500) == score(references(events), 500, task)


def test_requests_the_window_still_shows_are_counted() -> None:
    result = score(refs("aab"), 100)
    assert (result.requests, result.served) == (3, 1)


def test_an_unfinished_last_line_is_reported_not_silently_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "session.jsonl"
    lines = [event.model_dump_json(by_alias=True) for event in _log()]
    log.write_text("\n".join(lines) + '\n{"schema": "loomcore.ev', encoding="utf-8")
    main(["500", str(log)])
    seen = capsys.readouterr()
    assert "unfinished last line" in seen.err
    assert "requests, budget 500 tokens" in seen.out


def test_the_report_names_both_counts() -> None:
    text = report(score(refs("abca"), 2), 2, "mine")
    assert "mine: 4 requests, budget 2 tokens" in text
    assert "first asks (a miss for any policy): 3" in text
    assert "the context store's rule:    1 of 1" in text
    assert "furthest in future:          0 of 1" in text
    assert "requests the window still showed:    0" in text


def test_the_replay_drops_and_clears_what_the_drivers_window_does() -> None:
    # The replay copies `build_window`'s rule, so run the real one turn by turn on a
    # log of the same exchanges and compare what each has dropped and cleared.
    # Each file is asked for twice running, so every second request is a hit and the
    # driver's item numbers (1, 2, 3...) differ from positions in the string.
    specs: list[tuple[str, str, JsonObject, str]] = [
        (
            f"file-{n // 2 % 9}",
            "read_file",
            {"path": f"f{n // 2 % 9}.py"},
            "x" * (90 * (n // 2 % 9 + 1)),
        )
        for n in range(80)
    ]
    dones = [completed(f"t{n}", text) for n, (_, _, _, text) in enumerate(specs)]
    string = [
        ref_for(key, name, args, done)
        for (key, name, args, _), done in zip(specs, dones, strict=True)
    ]
    task = "Do the task. " * 60  # big enough to move what fits
    task_tokens = estimate_tokens([{"role": "user", "content": task}])
    budget = 1200
    replay = replay_policy(string, budget, task_tokens)

    events: list[Event] = []
    dropped: set[str] = set()
    cleared: set[str] = set()
    positions: dict[str, int] = {}
    for n, i in enumerate(replay.created):
        _, name, args, text = specs[i]
        use = dones[i].tool_use_id
        message: JsonObject = {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": use, "name": name, "input": args}],
            "stop_reason": "tool_use",
        }
        positions[f"seq:r:{2 * n}"] = i
        events.append(
            make_event(
                "model_call",
                seq=2 * n,
                timestamp=NOW,
                session_id="s",
                run_id="r",
                message=message,
                duration_ms=1,
                cost_usd=Decimal(0),
                pricing_version="v",
            )
        )
        events.append(
            make_event(
                "tool_completed",
                seq=2 * n + 1,
                timestamp=NOW,
                session_id="s",
                run_id="r",
                tool_use_id=use,
                ok=True,
                content=text,
                duration_ms=1,
                cached=False,
                from_tool=True,
            )
        )
        window = build_window(
            task,
            events,
            Carried(dropped, cleared),
            Budget(budget, 0.8, 0.6),
            estimate_tokens,
        )
        dropped |= set(window.newly_dropped)
        cleared |= set(window.newly_cleared)
        # The state every later decision is made on must agree after each window.
        assert {positions[i] for i in dropped} == replay.steps[n][0]
        assert {positions[i] for i in cleared} == replay.steps[n][1]

    assert len(replay.created) < len(specs), "some requests are hits, with no page"
    assert replay.dropped, "the budget is tight enough to drop something"
    assert replay.cleared, "and to clear something"
    assert {positions[i] for i in dropped} == replay.dropped
    assert {positions[i] for i in cleared} == replay.cleared
