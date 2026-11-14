"""Which tool calls of one turn must wait for which (spec section 3.2).

Pure: it takes what each call declares and returns an order, so it is tested with
no threads. Two calls conflict when either writes a resource the other reads or
writes. A call that is not `read_only` and declares no `writes` may write anything,
so it conflicts with every other call. A conflict is an edge from the earlier call
to the later one, in the order the model emitted them, so edges point forward and
there is no cycle. Resources are compared as strings, by equality.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Access:
    """What a tool declared at registration."""

    read_only: bool
    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()

    @property
    def writes_anything(self) -> bool:
        return not self.read_only and not self.writes


def conflict(a: Access, b: Access) -> bool:
    if a.writes_anything or b.writes_anything:
        return True
    return bool(a.writes & (b.reads | b.writes) or b.writes & (a.reads | a.writes))


def waits_for(calls: Sequence[Access]) -> list[frozenset[int]]:
    """For each call, the indexes of the earlier calls it has to wait for."""
    return [
        frozenset(j for j in range(i) if conflict(calls[j], call))
        for i, call in enumerate(calls)
    ]


def waves(calls: Sequence[Access]) -> list[list[int]]:
    """Indexes grouped so that the calls of a wave may run together and a call
    always comes in a later wave than every call it waits for."""
    level: list[int] = []
    for before in waits_for(calls):
        level.append(1 + max((level[j] for j in before), default=-1))
    return [
        [i for i, n in enumerate(level) if n == wave]
        for wave in range(max(level, default=-1) + 1)
    ]
