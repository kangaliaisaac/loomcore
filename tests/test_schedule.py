"""f15: which calls of a turn wait for which."""

from hypothesis import given
from hypothesis import strategies as st

from loomcore.schedule import Access, conflict, waits_for, waves

RESOURCES = st.frozensets(st.sampled_from(["a", "b", "c"]), max_size=2)
ACCESS = st.builds(Access, read_only=st.booleans(), reads=RESOURCES, writes=RESOURCES)
READ = Access(True, frozenset({"a"}))
WRITE_A = Access(False, writes=frozenset({"a"}))
WRITE_B = Access(False, writes=frozenset({"b"}))
UNDECLARED = Access(False)


def test_two_reads_do_not_conflict() -> None:
    assert not conflict(READ, READ)
    assert not conflict(Access(True), Access(True, frozenset({"a"})))


def test_a_write_conflicts_with_a_read_and_a_write_of_the_same_resource() -> None:
    assert conflict(WRITE_A, READ)
    assert conflict(READ, WRITE_A)
    assert conflict(WRITE_A, WRITE_A)
    assert not conflict(WRITE_A, WRITE_B)
    assert not conflict(WRITE_B, READ)


def test_a_tool_that_may_write_anything_conflicts_with_every_other_call() -> None:
    assert UNDECLARED.writes_anything
    for other in (READ, WRITE_B, Access(True), UNDECLARED):
        assert conflict(UNDECLARED, other)
        assert conflict(other, UNDECLARED)
    declared_reads_only = Access(False, reads=frozenset({"a"}))
    assert declared_reads_only.writes_anything  # reads do not make it narrower
    assert conflict(declared_reads_only, WRITE_B)


def test_conflicting_calls_wait_in_order_and_independent_ones_share_a_wave() -> None:
    calls = [READ, WRITE_B, WRITE_A, READ]
    assert waits_for(calls) == [
        frozenset(),
        frozenset(),
        frozenset({0}),
        frozenset({2}),
    ]
    assert waves(calls) == [[0, 1], [2], [3]]


def test_no_calls_means_no_waves() -> None:
    assert waves([]) == []


@given(st.lists(ACCESS, max_size=8))
def test_edges_point_forward_and_every_conflicting_pair_is_ordered(
    calls: list[Access],
) -> None:
    before = waits_for(calls)
    wave_of = {i: n for n, wave in enumerate(waves(calls)) for i in wave}
    assert sorted(wave_of) == list(range(len(calls)))  # every call is in one wave
    for i, earlier in enumerate(before):
        assert all(j < i for j in earlier)
        for j in range(i):
            if conflict(calls[j], calls[i]):
                assert j in earlier
                assert wave_of[j] < wave_of[i]


@given(ACCESS, ACCESS)
def test_conflict_is_symmetric(a: Access, b: Access) -> None:
    assert conflict(a, b) == conflict(b, a)
