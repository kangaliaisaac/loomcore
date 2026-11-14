"""Collecting what a tool call causes: refusals and notes (spec section 3.2).

The workspace and the sandbox report through plain callbacks fixed at construction,
which know nothing about the call that made them. A `Notices` is created first and
its two methods are passed as those callbacks; `collecting()` then opens a bucket
for one call, held in a `ContextVar`, so the notices land with the call that caused
them. A worker thread does not inherit the context: run it under
`contextvars.copy_context().run` and it shares the caller's bucket.
"""

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from loomcore.ports import Notice
from loomcore.sandbox import Refusal


class Notices:
    def __init__(self) -> None:
        # Per instance, so two registries' calls keep separate buckets.
        self._bucket: ContextVar[list[Notice] | None] = ContextVar(
            "loomcore_notices", default=None
        )

    @contextmanager
    def collecting(self) -> Iterator[list[Notice]]:
        """Notices raised inside the block (and in threads run under a copy of its
        context) are appended to the yielded list."""
        bucket: list[Notice] = []
        token = self._bucket.set(bucket)
        try:
            yield bucket
        finally:
            self._bucket.reset(token)

    def refused(self, refusal: Refusal) -> None:
        self._add(Notice("sandbox_denied", dataclasses.asdict(refusal)))

    def noted(self, path: str) -> None:
        self._add(Notice("note_written", {"path": path}))

    def _add(self, notice: Notice) -> None:
        bucket = self._bucket.get()
        if bucket is not None:  # outside a tool call there is no call to blame
            bucket.append(notice)
