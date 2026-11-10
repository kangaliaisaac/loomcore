"""f13: every path is resolved first and must stay inside the workspace root."""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from loomcore.sandbox import Refusal
from loomcore.workspace import (
    FilesystemWorkspace,
    PathRefusedError,
    Workspace,
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


def test_deleting_a_link_removes_the_link_and_not_what_it_points_to(
    root: Path,
) -> None:
    ws = Watched(root).workspace
    ws.write("real.txt", "kept")
    (root / "alias.txt").symlink_to(root / "real.txt")
    (root / "sub").mkdir()
    (root / "dirlink").symlink_to(root / "sub", target_is_directory=True)
    ws.delete("alias.txt")
    ws.delete("dirlink")
    assert ws.read("real.txt") == "kept"
    assert (root / "sub").is_dir()
    assert not (root / "alias.txt").is_symlink()
    assert not (root / "dirlink").is_symlink()


def test_a_link_out_or_a_dangling_link_can_be_removed_without_touching_the_target(
    root: Path,
) -> None:
    ws = Watched(root).workspace
    (root / "out.txt").symlink_to(root.parent / "outside" / "secret.txt")
    (root / "gone").symlink_to(root / "missing")  # dangling, inside
    ws.delete("out.txt")
    ws.delete("gone")
    assert (root.parent / "outside" / "secret.txt").read_text() == "secret"
    assert ws.list() == []


def test_deleting_inside_a_symlinked_parent_that_leaves_the_root_is_refused(
    root: Path,
) -> None:
    (root / "dir").symlink_to(root.parent / "outside", target_is_directory=True)
    watched = Watched(root)
    with pytest.raises(PathRefusedError):
        watched.workspace.delete("dir/secret.txt")
    assert (root.parent / "outside" / "secret.txt").read_text() == "secret"
    assert watched.refusals[0].target == "dir/secret.txt"


def test_a_path_with_a_nul_byte_is_refused_and_logged(root: Path) -> None:
    watched = Watched(root)
    for action in (watched.workspace.read, watched.workspace.list):
        with pytest.raises(PathRefusedError):
            action("a\x00b")
    with pytest.raises(PathRefusedError):
        watched.workspace.write("a\x00b", "x")
    before = watched.workspace.generation
    with pytest.raises(PathRefusedError):
        watched.workspace.delete("a\x00b")  # in the last component, not the parent
    assert watched.workspace.generation == before
    assert len(watched.refusals) == 4
    assert "cannot be resolved" in watched.refusals[0].reason


def test_listing_through_a_link_names_the_entries_as_asked_for(root: Path) -> None:
    ws = Watched(root).workspace
    ws.write("sub/x.txt", "x")
    (root / "alias").symlink_to(root / "sub", target_is_directory=True)
    assert ws.list("alias") == ["alias/x.txt"]


def test_a_notes_file_outside_the_root_is_refused_at_construction(
    root: Path,
) -> None:
    with pytest.raises(PathRefusedError):
        FilesystemWorkspace(root, notes="../elsewhere.md")


def test_a_link_to_the_notes_file_still_writes_the_notes(root: Path) -> None:
    watched = Watched(root)
    watched.workspace.write("notes.md", "first")
    (root / "same.md").symlink_to(root / "notes.md")
    watched.workspace.write("same.md", "second")
    assert watched.notes == ["notes.md", "notes.md"]  # the file written, by its name


SEGMENTS = st.sampled_from(["..", ".", "a", "b", "link", "dirlink", "/", ""])
OPERATIONS = st.sampled_from(["read", "write", "delete", "list"])


def snapshot(directory: Path, skip: Path | None = None) -> dict[str, str]:
    """Every file under `directory` with its content, and every directory; the
    workspace root itself is skipped, since it is meant to change."""
    return {
        str(entry.relative_to(directory)): entry.read_text() if entry.is_file() else ""
        for entry in sorted(directory.rglob("*"))
        if skip is None or not entry.is_relative_to(skip)
    }


@given(parts=st.lists(SEGMENTS, max_size=6), operation=OPERATIONS)
def test_nothing_outside_the_root_is_read_or_changed(
    tmp_path_factory: pytest.TempPathFactory, parts: list[str], operation: str
) -> None:
    """The oracle is what happens on disk, not what resolve_inside says."""
    base = tmp_path_factory.mktemp("ws")
    root, outside = base / "work", base / "outside"
    (root / "a").mkdir(parents=True)
    outside.mkdir()
    (outside / "f").write_text("secret")
    (outside / "g").write_text("other")
    (root / "link").symlink_to(outside / "f")
    (root / "dirlink").symlink_to(outside, target_is_directory=True)
    before = snapshot(base, skip=root)
    ws = FilesystemWorkspace(root)
    path = "/".join(parts)
    try:
        if operation == "read":
            assert ws.read(path) not in ("secret", "other")
        elif operation == "write":
            ws.write(path, "PWNED")
        elif operation == "delete":
            ws.delete(path)
        else:
            assert not {"f", "g"} & {Path(name).name for name in ws.list(path)}
    except PathRefusedError, OSError:
        pass  # refused, or an ordinary error such as a missing file
    assert snapshot(base, skip=root) == before
