"""f02 and f03: the writer, the reader, the lock and the torn-line rules."""

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from factories import make

from loomcore.eventlog import (
    LogCorruptError,
    SessionLockedError,
    open_writer,
    read_events,
)
from loomcore.events import dump_event


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "project" / "s1.jsonl"


def test_events_read_back_in_order(log: Path) -> None:
    events = [make("run_started"), make("turn_started", seq=1)]
    with open_writer(log) as writer:
        for event in events:
            writer.append(event)
    result = read_events(log)
    assert result.events == tuple(events)
    assert result.torn_tail is None


def test_a_writer_appends_to_what_is_already_there(log: Path) -> None:
    with open_writer(log) as writer:
        writer.append(make("run_started"))
    with open_writer(log) as writer:
        writer.append(make("turn_started", seq=1))
    assert [e.seq for e in read_events(log).events] == [0, 1]


def test_fsync_is_optional_and_used_when_asked(
    log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[int] = []
    monkeypatch.setattr(os, "fsync", synced.append)
    with open_writer(log) as writer:
        writer.append(make("run_started"))
    assert synced == []
    with open_writer(log, fsync=True) as writer:
        writer.append(make("turn_started", seq=1))
    assert len(synced) == 1


def test_a_torn_final_line_is_cut_before_the_next_append(log: Path) -> None:
    with open_writer(log) as writer:
        writer.append(make("run_started"))
    complete = log.read_bytes()
    log.write_bytes(complete + dump_event(make("turn_started", seq=1)).encode()[:20])
    with open_writer(log) as writer:
        writer.append(make("turn_started", seq=1))
    result = read_events(log)
    assert [e.seq for e in result.events] == [0, 1]
    assert result.torn_tail is None


def test_the_writer_keeps_the_torn_line_it_cut(log: Path) -> None:
    with open_writer(log) as writer:
        writer.append(make("run_started"))
        assert writer.cut_tail is None
    torn = dump_event(make("turn_started", seq=1)).encode()[:20]
    log.write_bytes(log.read_bytes() + torn)
    with open_writer(log) as writer:
        assert writer.cut_tail == torn


def test_a_failed_append_leaves_no_partial_line(
    log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = os.write

    def disk_full_after_ten_bytes(descriptor: int, data: bytes | memoryview) -> int:
        real_write(descriptor, bytes(data[:10]))
        raise OSError(28, "No space left on device")

    with open_writer(log) as writer:
        writer.append(make("run_started"))
        monkeypatch.setattr(os, "write", disk_full_after_ten_bytes)
        with pytest.raises(OSError, match="No space"):
            writer.append(make("turn_started", seq=1))
        monkeypatch.undo()
        writer.append(make("turn_started", seq=1))
    result = read_events(log)
    assert [e.seq for e in result.events] == [0, 1]
    assert result.torn_tail is None


def test_a_log_that_is_only_a_torn_line_is_emptied_by_the_writer(log: Path) -> None:
    log.parent.mkdir()
    log.write_bytes(b'{"schema": "loomc')
    with open_writer(log):
        pass
    assert log.read_bytes() == b""


def test_the_reader_drops_and_reports_a_torn_final_line(log: Path) -> None:
    with open_writer(log) as writer:
        writer.append(make("run_started"))
    log.write_bytes(log.read_bytes() + b'{"schema": "loomcore.ev')
    result = read_events(log)
    assert len(result.events) == 1
    assert result.torn_tail == b'{"schema": "loomcore.ev'


def test_a_broken_line_anywhere_else_is_an_error(log: Path) -> None:
    first, second = (
        dump_event(make("run_started")),
        dump_event(make("turn_started", seq=1)),
    )
    log.parent.mkdir()
    log.write_text(f"{first}\nnot json\n{second}\n")
    with pytest.raises(LogCorruptError, match="line 2"):
        read_events(log)


def test_a_complete_but_invalid_final_line_is_an_error_not_a_torn_write(
    log: Path,
) -> None:
    log.parent.mkdir()
    log.write_text('{"kind": "nonsense"}\n')
    with pytest.raises(LogCorruptError):
        read_events(log)


def test_a_second_writer_on_one_session_file_is_refused(log: Path) -> None:
    with open_writer(log), pytest.raises(SessionLockedError), open_writer(log):
        pass


def test_the_lock_is_released_when_the_writer_closes(log: Path) -> None:
    with open_writer(log):
        pass
    with open_writer(log):
        pass


def test_a_crash_while_holding_the_lock_does_not_block_the_next_writer(
    log: Path,
) -> None:
    holder = textwrap.dedent(
        """
        import sys, time
        from pathlib import Path
        from loomcore.eventlog import open_writer
        with open_writer(Path(sys.argv[1])):
            print("locked", flush=True)
            time.sleep(60)
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", holder, str(log)], stdout=subprocess.PIPE, text=True
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "locked"
    with pytest.raises(SessionLockedError), open_writer(log):
        pass
    process.send_signal(signal.SIGKILL)
    process.wait()
    with open_writer(log) as writer:
        writer.append(make("run_started"))


def test_a_line_that_is_not_utf8_is_corruption_not_a_crash(log: Path) -> None:
    log.parent.mkdir()
    log.write_bytes(b'\xff\xfe{"kind": 1}\n')
    with pytest.raises(LogCorruptError):
        read_events(log)


def test_an_interrupt_mid_append_leaves_no_partial_line(
    log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = os.write

    def interrupted_after_ten_bytes(descriptor: int, data: bytes | memoryview) -> int:
        real_write(descriptor, bytes(data[:10]))
        raise KeyboardInterrupt

    with open_writer(log) as writer:
        monkeypatch.setattr(os, "write", interrupted_after_ten_bytes)
        with pytest.raises(KeyboardInterrupt):
            writer.append(make("run_started"))
    assert log.read_bytes() == b""
