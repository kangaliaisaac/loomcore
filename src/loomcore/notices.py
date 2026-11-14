"""Collecting what a tool call causes: refusals and notes (spec section 3.2).

The workspace and the sandbox report through plain callbacks fixed at construction,
which know nothing about the call that made them. A `Notices` is created first and
its two methods are passed as those callbacks. The driver owns it (`Ports.notices`)
and wraps each tool call in `collecting()`, which opens a bucket held in a
`ContextVar`; the driver writes the bucket in a `finally`, so what a call caused is
logged whether the tool returns or breaks. Whether a worker thread inherits the
context depends on the interpreter build: run it under
`contextvars.copy_context().run` and it shares the caller's bucket. Nothing ties
the `Notices` given to the workspace and the sandbox to the driver's, so one that
reports with no bucket open logs a warning instead of dropping the notice quietly.
"""

import dataclasses
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from loomcore.events import JsonObject
from loomcore.sandbox import Refusal

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Notice:
    """An event a tool call caused (`sandbox_denied`, `note_written`); the driver
    adds the call's `tool_use_id` and writes it."""

    kind: Literal["sandbox_denied", "note_written"]
    fields: JsonObject


class Notices:
    def __init__(self) -> None:
        # Per instance, so two drivers' calls keep separate buckets.
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
        if bucket is None:  # outside a tool call there is no call to blame
            _LOG.warning(
                "%s reported outside a tool call and not logged: %s",
                notice.kind,
                notice.fields,
            )
            return
        bucket.append(notice)
