"""f13: every path is resolved first and must stay inside the workspace root."""

import os
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from loomcore.sandbox import Refusal
from loomcore.workspace import (
    FilesystemWorkspace,
    PathRefusedError,
    Workspace,
    resolve_inside,
)


class Watched:
    """A workspace with its refusals and notes recorded."""

    def __init__(self, root: Path) -> None:
        self.refusals: list[Refusal] = []
        self.notes: list[str] = []
        self.workspace: Workspace = FilesystemWorkspace(
            root, on_refusal=self.refusals.append, on_note=self.notes.append
        )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "work").mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("secret")
    return tmp_path / "work"


def test_a_file_can_be_written_read_listed_and_deleted(root: Path) -> None:
    ws = Watched(root).workspace
    ws.write("a/b.txt", "hello")  # parents are created inside the root
    assert ws.read("a/b.txt") == "hello"
    assert ws.list() == ["a"]
    assert ws.list("a") == ["a/b.txt"]
    ws.delete("a/b.txt")
    assert ws.list("a") == []


def test_dot_dot_is_refused_and_logged(root: Path) -> None:
    watched = Watched(root)
    with pytest.raises(PathRefusedError) as raised:
        watched.workspace.read("../outside/secret.txt")
    assert watched.refusals == [raised.value.refusal]
    assert (watched.refusals[0].action, watched.refusals[0].target) == (
        "read",
        "../outside/secret.txt",
    )
    assert "outside the workspace" in watched.refusals[0].reason


def test_an_absolute_path_outside_is_refused_and_one_inside_is_accepted(
    root: Path,
) -> None:
    watched = Watched(root)
    with pytest.raises(PathRefusedError):
        watched.workspace.read(str(root.parent / "outside" / "secret.txt"))
    watched.workspace.write("here.txt", "x")
    assert watched.workspace.read(str(root.resolve() / "here.txt")) == "x"
    assert len(watched.refusals) == 1


def test_a_symlink_out_of_the_workspace_is_refused(root: Path) -> None:
    (root / "link.txt").symlink_to(root.parent / "outside" / "secret.txt")
    watched = Watched(root)
    with pytest.raises(PathRefusedError):
        watched.workspace.read("link.txt")
    with pytest.raises(PathRefusedError):
        watched.workspace.write("link.txt", "overwritten")
    assert (root.parent / "outside" / "secret.txt").read_text() == "secret"


def test_a_symlinked_parent_directory_is_refused(root: Path) -> None:
    (root / "dir").symlink_to(root.parent / "outside", target_is_directory=True)
    watched = Watched(root)
    with pytest.raises(PathRefusedError):
        watched.workspace.read("dir/secret.txt")
    with pytest.raises(PathRefusedError):
        watched.workspace.write("dir/new.txt", "x")  # a file that does not exist yet
    assert not (root.parent / "outside" / "new.txt").exists()
    with pytest.raises(PathRefusedError):
        watched.workspace.list("dir")


def test_a_dangling_symlink_out_is_refused(root: Path) -> None:
    (root / "dangling").symlink_to(root.parent / "outside" / "not-yet")
    with pytest.raises(PathRefusedError):
        Watched(root).workspace.write("dangling", "x")
    assert not (root.parent / "outside" / "not-yet").exists()


def test_a_symlink_that_stays_inside_is_followed(root: Path) -> None:
    ws = Watched(root).workspace
    ws.write("real.txt", "inside")
    (root / "alias.txt").symlink_to(root / "real.txt")
    assert ws.read("alias.txt") == "inside"


def test_a_symlinked_root_works(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    ws = Watched(link).workspace  # like /tmp on macOS
    ws.write("a.txt", "x")
    assert ws.read("a.txt") == "x"
    assert (real / "a.txt").exists()


def test_the_root_cannot_be_deleted(root: Path) -> None:
    watched = Watched(root)
    with pytest.raises(PathRefusedError, match="root cannot be deleted"):
        watched.workspace.delete(".")
    assert root.exists()
    assert watched.refusals[0].action == "delete"


def test_the_generation_counts_every_write_and_delete_and_bump(root: Path) -> None:
    ws = Watched(root).workspace
    assert ws.generation == 0
    ws.write("a.txt", "1")
    assert ws.generation == 1
    ws.delete("a.txt")
    assert ws.generation == 2
    ws.bump()  # a write the workspace cannot see
    assert ws.generation == 3


def test_reads_and_refusals_leave_the_generation_alone(root: Path) -> None:
    ws = Watched(root).workspace
    ws.write("a.txt", "1")
    before = ws.generation
    ws.read("a.txt")
    ws.list()
    with pytest.raises(PathRefusedError):
        ws.write("../escape.txt", "x")
    with pytest.raises(PathRefusedError):
        ws.delete("../outside/secret.txt")
    assert ws.generation == before


def test_a_failed_write_still_bumps_because_it_may_have_changed_the_file(
    root: Path,
) -> None:
    ws = Watched(root).workspace
    ws.write("d/f.txt", "1")
    before = ws.generation
    with pytest.raises(IsADirectoryError):
        ws.write("d", "x")  # d is a directory
    assert ws.generation == before + 1


def test_a_write_to_the_notes_file_is_reported_and_others_are_not(
    root: Path,
) -> None:
    watched = Watched(root)
    watched.workspace.write("other.md", "x")
    assert watched.notes == []
    watched.workspace.write("notes.md", "remember")
    watched.workspace.write("./sub/../notes.md", "again")
    assert watched.notes == ["notes.md", "notes.md"]


def test_the_notes_file_is_the_one_asked_for(root: Path) -> None:
    notes: list[str] = []
    ws = FilesystemWorkspace(root, notes="memory/keep.md", on_note=notes.append)
    ws.write("notes.md", "x")
    ws.write("memory/keep.md", "y")
    assert notes == ["memory/keep.md"]


def test_listing_a_file_or_reading_a_missing_one_is_an_ordinary_error(
    root: Path,
) -> None:
    ws = Watched(root).workspace
    ws.write("f.txt", "x")
    with pytest.raises(NotADirectoryError):
        ws.list("f.txt")
    with pytest.raises(FileNotFoundError):
        ws.read("missing.txt")


SEGMENTS = st.sampled_from(["..", ".", "a", "b", "link", "dirlink", "/", ""])


@given(parts=st.lists(SEGMENTS, max_size=6))
def test_no_path_resolves_outside_the_root(
    tmp_path_factory: pytest.TempPathFactory, parts: list[str]
) -> None:
    base = tmp_path_factory.mktemp("ws")
    root = (base / "work").resolve()
    (root / "a").mkdir(parents=True, exist_ok=True)
    (base / "outside").mkdir(exist_ok=True)
    for name, target in (
        ("link", base / "outside" / "f"),
        ("dirlink", base / "outside"),
    ):
        if not os.path.lexists(root / name):
            (root / name).symlink_to(target)
    path = "/".join(parts)
    try:
        resolved = resolve_inside(root, path)
    except PathRefusedError:
        return
    assert resolved.is_relative_to(root)
