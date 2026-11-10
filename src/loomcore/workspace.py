"""The filesystem workspace: the only place the agent reads and writes (spec 3.4).

Every path goes through `resolve_inside`, which resolves first (following symlinks)
and then requires the result to be inside the root. The check is not atomic with
the open that follows: a process that can race the workspace can still swap a
symlink in between, so the sandbox (a subprocess with no such access), not this
class, is what contains hostile code.
"""

from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, Protocol

from loomcore.sandbox import Refusal


class PathRefusedError(Exception):
    """A path the workspace will not touch; `refusal` is what gets logged."""

    def __init__(self, refusal: Refusal) -> None:
        super().__init__(f"{refusal.action} {refusal.target}: {refusal.reason}")
        self.refusal = refusal


def resolve_inside(root: Path, path: str) -> Path:
    """`path` as an absolute path under `root`, or `PathRefusedError`.

    `root` must already be resolved. An absolute path is accepted only when it
    resolves inside the root; `..` and symlinks are followed before the check, so a
    link out, or a symlinked parent, leads outside and is refused. The path need not
    exist (a write creates it), and a dangling link is followed as far as it goes.
    """
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        reason = "the path resolves outside the workspace"
        raise PathRefusedError(Refusal("resolve", path, reason))
    return resolved


class Workspace(Protocol):
    def read(self, path: str) -> str: ...
    def write(self, path: str, content: str) -> None: ...
    def list(self, path: str = ".") -> list[str]: ...
    def delete(self, path: str) -> None: ...
    @property
    def generation(self) -> int: ...  # incremented on every write or delete
    def bump(self) -> None: ...  # for writes the workspace cannot see


def _ignore(_value: object) -> None:
    return None


class FilesystemWorkspace:
    """A `Workspace` on a directory.

    `on_refusal` is called with every refusal before the error is raised (the
    driver turns it into `sandbox_denied`); `on_note` with the path of every write
    to the notes file (it becomes `note_written`).
    """

    def __init__(
        self,
        root: Path,
        *,
        notes: str = "notes.md",
        on_refusal: Callable[[Refusal], None] = _ignore,
        on_note: Callable[[str], None] = _ignore,
    ) -> None:
        # Resolved once, so a symlinked root such as /tmp on macOS does not make
        # every path look like an escape.
        self._root = root.resolve()
        self._notes = (self._root / notes).resolve()
        self._on_refusal = on_refusal
        self._on_note = on_note
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def bump(self) -> None:
        self._generation += 1

    def read(self, path: str) -> str:
        return self._resolve("read", path).read_text()

    def list(self, path: str = ".") -> list[str]:
        directory = self._resolve("list", path)
        return sorted(self._relative(child) for child in directory.iterdir())

    def write(self, path: str, content: str) -> None:
        target = self._resolve("write", path)
        try:  # bumped even if the write fails: it may have changed the file
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        finally:
            self.bump()
        if target == self._notes:
            self._on_note(self._relative(target))

    def delete(self, path: str) -> None:
        target = self._resolve("delete", path)
        if target == self._root:
            self._refuse("delete", path, "the workspace root cannot be deleted")
        try:
            if target.is_dir():
                target.rmdir()
            else:
                target.unlink()
        finally:
            self.bump()

    def _resolve(self, action: str, path: str) -> Path:
        try:
            return resolve_inside(self._root, path)
        except PathRefusedError as error:
            self._refuse(action, path, error.refusal.reason)

    def _refuse(self, action: str, path: str, reason: str) -> NoReturn:
        refusal = Refusal(action, path, reason)
        self._on_refusal(refusal)
        raise PathRefusedError(refusal)

    def _relative(self, path: Path) -> str:
        return path.relative_to(self._root).as_posix()
