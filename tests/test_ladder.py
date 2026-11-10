"""f11: the repeat detector is a pure judgement, in emission order."""

from loomcore.ladder import tripped


def test_a_call_trips_when_the_window_holds_the_threshold_of_it() -> None:
    assert tripped([], ["a", "a", "a"], window=10, threshold=3) == [2]


def test_different_calls_never_trip() -> None:
    assert tripped(["a", "b"], ["c", "a", "b"], window=10, threshold=2) == [1, 2]


def test_the_runs_earlier_calls_count() -> None:
    assert tripped(["a", "a"], ["a"], window=10, threshold=3) == [0]


def test_a_call_older_than_the_window_does_not_count() -> None:
    history = ["a", "a", "x", "y"]
    assert tripped(history, ["a"], window=3, threshold=2) == []
    assert tripped(history, ["a"], window=4, threshold=2) == [0]  # a, x, y, a
    assert tripped(history, ["a"], window=5, threshold=3) == [0]  # a, a, x, y, a


def test_each_call_is_judged_against_the_calls_up_to_itself() -> None:
    assert tripped([], ["a", "a", "a", "a"], window=10, threshold=2) == [1, 2, 3]


def test_several_calls_can_trip_in_one_turn() -> None:
    assert tripped(["a", "b"], ["a", "b"], window=10, threshold=2) == [0, 1]
