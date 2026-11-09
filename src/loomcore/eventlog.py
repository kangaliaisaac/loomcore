"""Reading and writing a session's JSONL event file (spec section 2).

This is the edge: files, locks and fsync. The events themselves are in events.py.
"""

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pydantic import ValidationError

from loomcore.events import Event, dump_event, parse_event

_CHUNK = 64 * 1024


class SessionLockedError(Exception):
    """Another process is writing this session file; retry when it has finished."""


class LogCorruptError(Exception):
    """A line before the end of the log does not parse; the file needs repair."""


@dataclass(frozen=True)
class ReadResult:
    events: tuple[Event, ...]
    torn_tail: bytes | None  # the unterminated final line a crash left, if any


def read_events(path: Path) -> ReadResult:
    """Read every event. An unterminated final line is dropped and reported."""
    *lines, tail = path.read_bytes().split(b"\n")
    events = []
    for number, line in enumerate(lines, start=1):
        try:
            events.append(parse_event(line))
        except ValidationError as error:
            message = f"{path} line {number} is not an event: {error}"
            raise LogCorruptError(message) from error
    return ReadResult(tuple(events), tail or None)


class EventWriter:
    """Appends events to one session file. Get one from `open_writer`."""

    def __init__(
        self, file: BinaryIO, *, fsync: bool, cut_tail: bytes | None = None
    ) -> None:
        self._file = file
        self._fsync = fsync
        self.cut_tail = cut_tail  # the torn final line `open_writer` cut off, if any

    def append(self, event: Event) -> None:
        """Write one line. If the write fails part-way, cut the partial line back."""
        descriptor = self._file.fileno()
        start = os.fstat(descriptor).st_size
        pending = memoryview(dump_event(event).encode() + b"\n")
        try:
            while pending:
                pending = pending[os.write(descriptor, pending) :]
        except OSError:
            os.ftruncate(descriptor, start)
            raise
        if self._fsync:
            os.fsync(descriptor)


@contextmanager
def open_writer(path: Path, *, fsync: bool = False) -> Iterator[EventWriter]:
    """Take the session lock, cut off any torn final line, and yield a writer.

    The lock is `flock` on `<sessionId>.lock`: the operating system drops it when
    the holder dies, so a crash never leaves a stale lock. The bytes it cut are on
    the writer as `cut_tail`, so a resume can report them. Raises
    `SessionLockedError` if another process holds it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SessionLockedError(
                f"{path} has another writer; retry later"
            ) from error
        with path.open("a+b") as file:
            complete = _complete_length(file)
            file.seek(complete)
            cut = file.read() or None
            file.truncate(complete)
            yield EventWriter(file, fsync=fsync, cut_tail=cut)


def _complete_length(file: BinaryIO) -> int:
    """The size up to and including the last newline, so a torn tail is cut."""
    position = file.seek(0, os.SEEK_END)
    while position > 0:
        start = max(0, position - _CHUNK)
        file.seek(start)
        newline = file.read(position - start).rfind(b"\n")
        if newline != -1:
            return start + newline + 1
        position = start
    return 0
