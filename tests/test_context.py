"""f16, part 1: packing and compaction are pure choices among counted items."""

import random

from hypothesis import given
from hypothesis import strategies as st

from loomcore.context import CLEARED, Cleared, Item, Packed, compact, pack
from loomcore.events import JsonObject


def item(
    k: int, tokens: int, *, pinned: bool = False, cleared_tokens: int | None = None
) -> Item:
    message: JsonObject = {"role": "user", "content": f"item {k}"}
    cleared = None
    if cleared_tokens is not None:
        blank: JsonObject = {"role": "user", "content": CLEARED}
        cleared = Cleared((blank,), cleared_tokens)
    return Item(f"i{k}", "exchange", k, tokens, (message,), pinned, cleared)


ITEM_SPECS = st.lists(
    st.tuples(
        st.integers(0, 40),
        st.booleans(),
        st.one_of(st.none(), st.integers(0, 10)),
    ),
    max_size=12,
)


def build(specs: list[tuple[int, bool, int | None]]) -> tuple[list[Item], list[Item]]:
    """(candidates, pinned) from specs of (tokens, pinned, cleared tokens)."""
    items = [
        item(k, t, pinned=p, cleared_tokens=c) for k, (t, p, c) in enumerate(specs)
    ]
    return [i for i in items if not i.pinned], [i for i in items if i.pinned]


def ids(items: tuple[Item, ...] | list[Item]) -> list[str]:
    return [i.id for i in items]


@given(ITEM_SPECS, st.integers(0, 120))
def test_packing_respects_the_budget_and_always_includes_the_pinned_items(
    specs: list[tuple[int, bool, int | None]], budget: int
) -> None:
    candidates, pinned = build(specs)
    packed = pack(candidates, budget, pinned)
    assert packed.tokens == sum(i.tokens for i in packed.selected)
    assert set(ids(pinned)) <= set(ids(packed.selected))
    if not packed.over:
        assert packed.tokens <= budget
    assert sorted(ids(packed.selected) + ids(packed.dropped)) == sorted(
        ids(candidates) + ids(pinned)
    )
    orders = [i.order for i in packed.selected]
    assert orders == sorted(orders)  # history order, whatever the choice order


@given(ITEM_SPECS, st.integers(0, 120), st.randoms())
def test_the_result_does_not_depend_on_the_order_the_candidates_come_in(
    specs: list[tuple[int, bool, int | None]], budget: int, shuffler: random.Random
) -> None:
    candidates, pinned = build(specs)
    shuffled = list(candidates)
    shuffler.shuffle(shuffled)
    assert pack(candidates, budget, pinned) == pack(shuffled, budget, pinned)


def test_pinned_items_that_do_not_fit_are_kept_and_flagged() -> None:
    pinned = [item(0, 30, pinned=True), item(1, 30, pinned=True)]
    packed = pack([item(2, 5), item(3, 5)], 40, pinned)
    assert packed.over
    assert ids(packed.selected) == ["i0", "i1"]
    assert ids(packed.dropped) == ["i2", "i3"]


def test_a_later_item_beats_an_earlier_one_of_the_same_size() -> None:
    packed = pack([item(1, 10), item(2, 10)], 10, [])
    assert ids(packed.selected) == ["i2"]
    assert ids(packed.dropped) == ["i1"]


def test_a_small_old_item_can_beat_a_large_recent_one_per_token() -> None:
    packed = pack([item(1, 2), item(2, 50)], 10, [])
    assert ids(packed.selected) == ["i1"]  # (1+1)/2 > (2+1)/50


def test_an_item_that_does_not_fit_is_skipped_and_smaller_ones_still_chosen() -> None:
    packed = pack([item(1, 8), item(2, 9), item(3, 3)], 11, [])
    assert ids(packed.selected) == ["i1", "i3"]  # i2 (9) leaves 2 < 3; i3 then i1


def test_items_of_no_tokens_are_always_in() -> None:
    packed = pack([item(1, 0), item(2, 0)], 0, [])
    assert ids(packed.selected) == ["i1", "i2"]


def window(*tokens: int, pinned_last: bool = True) -> Packed:
    items = [
        item(k, t, pinned=pinned_last and k == len(tokens) - 1, cleared_tokens=1)
        for k, t in enumerate(tokens)
    ]
    return pack(
        [i for i in items if not i.pinned], 1000, [i for i in items if i.pinned]
    )


def test_a_window_below_the_threshold_is_left_alone() -> None:
    packed = window(100, 100, 100)  # 300 of 1000
    assert compact(packed, 0.8, 0.6) is packed


def test_at_the_threshold_the_oldest_results_are_cleared_down_to_the_target() -> None:
    packed = window(300, 300, 300, 300)  # 1200 would not fit; budget keeps 1000
    packed = pack(list(packed.selected), 1000, [])
    packed = window(250, 250, 250, 100)  # 850 of 1000, over 0.8
    compacted = compact(packed, 0.8, 0.6)
    assert compacted.cleared == ("i0", "i1")  # 850 -> 602 -> 354: stops at <= 600
    assert compacted.tokens == sum(i.tokens for i in compacted.selected)
    assert compacted.tokens <= 0.6 * 1000
    assert [i.messages[0]["content"] for i in compacted.selected][:2] == [CLEARED] * 2
    assert ids(compacted.selected) == ids(packed.selected)  # nothing dropped


def test_a_pinned_item_is_never_cleared() -> None:
    packed = window(500, 400, pinned_last=True)  # the second is pinned
    compacted = compact(packed, 0.5, 0.1)
    assert compacted.cleared == ("i0",)
    assert compacted.selected[1].tokens == 400
    assert compacted.selected[1].messages[0]["content"] == "item 1"


def test_a_window_with_nothing_to_clear_comes_back_as_it_is() -> None:
    bare = pack([item(0, 700), item(1, 200)], 1000, [])
    assert compact(bare, 0.8, 0.6) == bare


def test_compaction_does_nothing_to_a_window_that_is_over() -> None:
    over = pack([], 10, [item(0, 50, pinned=True)])
    assert over.over
    assert compact(over, 0.0, 0.0) is over


@given(ITEM_SPECS, st.integers(1, 200))
def test_compaction_keeps_the_books_straight_and_is_idempotent(
    specs: list[tuple[int, bool, int | None]], budget: int
) -> None:
    candidates, pinned = build(specs)
    once = compact(pack(candidates, budget, pinned), 0.8, 0.6)
    assert once.tokens == sum(i.tokens for i in once.selected)
    assert compact(once, 0.8, 0.6) == once or once.tokens > 0.6 * budget
    for cleared_id in once.cleared:
        assert cleared_id not in ids(pinned)
