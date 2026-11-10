"""The sandbox boundary (spec section 3.3): every command runs in a Docker container.

`evaluate` is the pure rule check. `docker_args` is the pure builder of the one
`docker run` command line, so every security flag can be tested without Docker.
`DockerSandbox` runs a permitted command in a fresh container: no network, a
read-only root filesystem, no capabilities, a non-root user, the workspace as the
only writable mount, only the environment variables the caller and the rules allow,
and limits on time, memory, processes, CPU, file size and output.

On macOS, Docker Desktop runs the container in a Linux virtual machine, so the
command sits behind a hypervisor as well as the container; on Linux it shares the
host kernel. Docker is the isolation: if the daemon or the image is missing the
sandbox is not built, and nothing falls back to running on the host.
"""

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import IO, NoReturn, Protocol

WORKSPACE = "/workspace"  # where the workspace is mounted in the container
LABEL = "loomcore-sandbox"  # on every container, so a stray one can be found
_NOT_RUNNABLE = 127  # the shell's code for a command that could not be started
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*\Z")  # never starts with "-"
_POLL_SECONDS = 0.02
_STOP_SECONDS = 20.0  # how long to keep removing a container before killing its CLI
_DOCKER_CALL_SECONDS = 30.0
_HOST_ONLY = (Path("/proc"), Path("/sys"), Path("/dev"))  # never mounted from the host
_KILLED = 137
_USER = re.compile(r"[1-9][0-9]*:[1-9][0-9]*\Z")  # uid:gid, neither root
_ENV_LINE_BYTES = 60 * 1024  # Docker reads an env-file line of at most 64 KB


@dataclass(frozen=True)
class Refusal:
    """What `sandbox_denied` logs: the action tried, its target, and why not."""

    action: str
    target: str
    reason: str


@dataclass(frozen=True)
class Permit:
    pass


@dataclass(frozen=True)
class Rules:
    """What is allowed; everything else is refused."""

    root: Path  # the workspace: mounted at /workspace; a cwd must be inside it
    image: str  # the image commands run in; it must already be present locally
    commands: frozenset[str]  # absolute paths, inside the image, that may run
    env: frozenset[str] = frozenset()  # variables that may reach a command
    read: frozenset[Path] = frozenset()  # extra host trees, mounted read-only


@dataclass(frozen=True)
class Request:
    command: tuple[str, ...]
    cwd: Path  # resolved, absolute, on the host
    env_allow: frozenset[str]


@dataclass(frozen=True)
class ResourceLimits:
    """Per command."""

    timeout: float = 30.0  # wall-clock seconds, then the container is killed
    cpu_seconds: int = 30  # CPU time of each process, not of the container
    memory_bytes: int = 512 * 1024 * 1024  # swap included
    pids: int = 128
    cpus: float = 1.0
    max_file_bytes: int = 64 * 1024 * 1024  # any one file it writes
    max_open_files: int = 256
    max_output_bytes: int = 1024 * 1024  # stdout and stderr together, then killed


@dataclass(frozen=True)
class ProcessResult:
    """`returncode` is the command's, as Docker reports it: 137 after a kill (the
    timeout always, whatever the docker client then reports; the output cap and the
    memory limit as Docker reports them), 127 when the command could not be started, 125
    when Docker itself failed."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    output_truncated: bool = False


def unusable(text: str, *, strict: bool = False) -> bool:
    """Text the operating system cannot take in a path or an argument: a NUL byte, or
    characters that cannot be encoded (a lone surrogate, which a model's JSON can
    carry). `strict` also refuses the surrogate-escape range (U+DC80 to U+DCFF), which
    the operating system takes as raw bytes but Docker's JSON turns into U+FFFD, so
    what reaches the container would differ from what was asked, silently."""
    try:
        text.encode("utf-8") if strict else os.fsencode(text)
    except ValueError:
        return True
    return "\x00" in text


def evaluate(rules: Rules, request: Request) -> Permit | Refusal:
    """Permit a request only if every rule holds; the first that fails is the reason."""
    target = " ".join(request.command)
    for reason in (
        _command_problem(rules, request),
        _cwd_problem(rules, request),
        _env_problem(rules, request),
    ):
        if reason is not None:
            return Refusal("exec", target, reason)
    return Permit()


def _command_problem(rules: Rules, request: Request) -> str | None:
    if not request.command:
        return "there is no command to run"
    if any(unusable(part, strict=True) for part in request.command):
        return "an argument contains a NUL byte or text that cannot be encoded"
    program = request.command[0]
    if not PurePath(program).is_absolute():
        return "give the absolute path of an allowed command, not a name to search for"
    return None if program in rules.commands else f"{program} is not an allowed command"


def _cwd_problem(rules: Rules, request: Request) -> str | None:
    if unusable(str(request.cwd), strict=True):
        return "the working directory has a NUL byte or text that cannot be encoded"
    if not request.cwd.is_absolute() or not request.cwd.is_relative_to(rules.root):
        return f"the working directory {request.cwd} is outside the workspace"
    return None


def _env_problem(rules: Rules, request: Request) -> str | None:
    extra = sorted(request.env_allow - rules.env)
    if extra:
        return f"the environment variable(s) {', '.join(extra)} are not allowed"
    bad = sorted(name for name in request.env_allow if not _NAME.match(name))
    if bad:
        return f"{', '.join(bad)} is not a variable name"
    return None


class CommandRefusedError(Exception):
    """A command the rules do not permit; `refusal` is what gets logged."""

    def __init__(self, refusal: Refusal) -> None:
        super().__init__(f"{refusal.action} {refusal.target}: {refusal.reason}")
        self.refusal = refusal


class SandboxUnavailableError(Exception):
    """Docker or the image is not there. The sandbox is not built, and nothing runs
    on the host instead."""


class Sandbox(Protocol):
    def run(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path,
        env_allow: Set[str],
        limits: ResourceLimits,
    ) -> ProcessResult: ...


@dataclass(frozen=True)
class Launch:
    """What differs between two runs of the same sandbox."""

    docker: str  # the docker executable
    name: str  # a fresh container name, so it can be killed and removed
    env_file: Path  # the allowed variables, mode 0600
    user: str  # "uid:gid", never root


def docker_args(
    rules: Rules, request: Request, limits: ResourceLimits, launch: Launch
) -> list[str]:
    """The whole `docker run` command line. Pure, so the flags can be tested.

    Nothing here is configurable by the caller of `run`: the isolation is fixed. The
    workspace is the only writable mount; extra trees are read-only; the root
    filesystem is read-only with a small `noexec` /tmp; there is no network, no
    capability, no new privilege, and the user is not root (`ValueError` otherwise).
    """
    if not _USER.match(launch.user):
        raise ValueError(f"{launch.user!r} is not a non-root uid:gid")
    for field, number in vars(limits).items():
        if not math.isfinite(number) or number <= 0:  # Docker reads 0 and -1 as "none"
            raise ValueError(
                f"limit {field}={number!r} must be a finite number above 0"
            )
    cwd = request.cwd.relative_to(rules.root).as_posix()
    memory = str(limits.memory_bytes)
    args = [
        launch.docker, "run", "--rm", "--pull", "never", "--init",
        "--name", launch.name, "--label", LABEL,
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--ipc", "none",  # no /dev/shm: the only writable places are listed below
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", launch.user,
        "--pids-limit", str(limits.pids),
        "--memory", memory, "--memory-swap", memory, "--cpus", str(limits.cpus),
        "--ulimit", f"cpu={limits.cpu_seconds}:{limits.cpu_seconds}",
        "--ulimit", f"fsize={limits.max_file_bytes}:{limits.max_file_bytes}",
        "--ulimit", f"nofile={limits.max_open_files}:{limits.max_open_files}",
        "--ulimit", "core=0:0",
        "--mount", f"type=bind,src={rules.root},dst={WORKSPACE}",
    ]  # fmt: skip
    for tree in sorted(rules.read):
        args += ["--mount", f"type=bind,src={tree},dst={tree},readonly"]
    workdir = WORKSPACE if cwd == "." else f"{WORKSPACE}/{cwd}"
    args += ["--workdir", workdir, "--env-file", str(launch.env_file)]
    # --entrypoint: an image's own ENTRYPOINT would otherwise run in place of the
    # allowed program and take it as an argument, so `commands` would allow nothing.
    program, *arguments = request.command
    return [*args, "--entrypoint", program, rules.image, *arguments]


def env_value_problem(name: str, value: str) -> str | None:
    """Why a value cannot go in an `--env-file`, or None. Docker reads one line per
    variable: it would change a trailing carriage return and refuse a long line."""
    if any(c in value for c in "\n\r\x00"):
        return (
            f"the value of {name} cannot be passed (a newline, carriage return or NUL)"
        )
    try:
        line = f"{name}={value}".encode()
    except UnicodeEncodeError:
        return f"the value of {name} cannot be passed (it is not valid UTF-8)"
    if len(line) > _ENV_LINE_BYTES:
        return f"the value of {name} cannot be passed (it is too long)"
    return None


def write_env_file(directory: Path, env: Mapping[str, str]) -> Path:
    """The variables as an `--env-file`, readable by its owner only. They go in a file
    and never on a command line, where a process listing would show them."""
    path = directory / "env"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        for name, value in env.items():
            file.write(f"{name}={value}\n")
    return path


class DockerSandbox:
    """A `Sandbox` that runs each command in its own container.

    The environment of the command is exactly the allowed variables the parent has
    (`environ`, the process environment by default) plus what the image itself
    defines (`PATH` and the like). `on_refusal` is called with every refusal before
    the error is raised; it has no default, because a sandbox whose refusals nobody
    hears would be a silent one.
    """

    def __init__(
        self,
        rules: Rules,
        *,
        on_refusal: Callable[[Refusal], None],
        environ: Mapping[str, str] | None = None,
        docker: str = "docker",
    ) -> None:
        self._rules = _checked(rules)
        self._user = _non_root_user()
        self._docker = _find_docker(docker)
        self._check_no_proxies()
        self._check_image()
        self._environ = os.environ if environ is None else environ
        self._on_refusal = on_refusal

    def run(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path,
        env_allow: Set[str],
        limits: ResourceLimits = ResourceLimits(),  # noqa: B008  (frozen)
    ) -> ProcessResult:
        command = tuple(cmd)
        try:  # relative to the workspace, like the workspace's own paths
            directory = (self._rules.root / cwd).resolve()
        except OSError, ValueError, RuntimeError:
            reason = "the working directory cannot be resolved"
            self._refuse(Refusal("exec", " ".join(command), reason))
        request = Request(command, directory, frozenset(env_allow))
        verdict = evaluate(self._rules, request)
        if isinstance(verdict, Refusal):
            self._refuse(verdict)
        if not directory.is_dir():  # like a missing executable: a result, not a crash
            reason = f"cannot run: {directory} is not a directory\n"
            return ProcessResult(_NOT_RUNNABLE, "", reason, timed_out=False)
        env = {n: self._environ[n] for n in sorted(env_allow) if n in self._environ}
        for name, value in env.items():
            if (problem := env_value_problem(name, value)) is not None:
                self._refuse(Refusal("exec", " ".join(command), problem))
        return self._execute(request, env, limits)

    def _execute(
        self, request: Request, env: dict[str, str], limits: ResourceLimits
    ) -> ProcessResult:
        name = f"loomcore-{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory(prefix="loomcore-env-") as private:
            Path(private).chmod(0o700)
            env_file = write_env_file(Path(private), env)
            launch = Launch(self._docker, name, env_file, self._user)
            args = docker_args(self._rules, request, limits, launch)
            return self._run_container(args, name, limits)

    def _run_container(
        self, args: list[str], name: str, limits: ResourceLimits
    ) -> ProcessResult:
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            process = subprocess.Popen(
                args, stdin=subprocess.DEVNULL, stdout=out, stderr=err
            )
            timed_out = False
            started = time.monotonic()
            try:
                while process.poll() is None:
                    timed_out = time.monotonic() - started > limits.timeout
                    over = _size(out) + _size(err) > limits.max_output_bytes
                    if timed_out or over:
                        break
                    time.sleep(_POLL_SECONDS)
            finally:
                self._stop(name, process)  # also if we were interrupted
            stdout, stderr, cut = _read_together(out, err, limits.max_output_bytes)
            code = process.returncode
            if timed_out or code < 0:  # a kill is a kill, however the client died
                code = _KILLED
            return ProcessResult(code, stdout, stderr, timed_out, cut)

    def _stop(self, name: str, process: subprocess.Popen[bytes]) -> None:
        """End the container and the docker client, leaving nothing behind.

        `docker kill` only reaches a running container, and Docker takes a while to
        start one, so a timeout can arrive first. `rm --force` kills a running
        container and removes one that was created but never started, so it is
        repeated until the client exits; the client is killed last, and the
        container is removed once more in case it appeared as the client died.
        """
        deadline = time.monotonic() + _STOP_SECONDS
        while process.poll() is None and time.monotonic() < deadline:
            self._remove(name)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=_POLL_SECONDS * 10)
        if process.poll() is None:
            process.kill()
            process.wait()
        self._remove(name)

    def _remove(self, name: str) -> None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            subprocess.run(
                [self._docker, "rm", "--force", name],
                capture_output=True,
                check=False,
                timeout=_DOCKER_CALL_SECONDS,
            )

    def _check_no_proxies(self) -> None:
        """The Docker CLI adds `proxies` from its config to every container it
        creates, which would put variables nobody allow-listed in the command's
        environment (often with credentials)."""
        data = _docker_config()
        if data.get("proxies"):
            raise SandboxUnavailableError(
                f"{_docker_config_dir() / 'config.json'} sets `proxies`, which Docker "
                "would pass into every container as environment variables. Remove it, "
                "or point DOCKER_CONFIG at a directory without it."
            )

    def _check_image(self) -> None:
        try:
            probe = subprocess.run(
                [self._docker, "image", "inspect", self._rules.image],
                capture_output=True,
                check=False,
                timeout=_DOCKER_CALL_SECONDS,
            )
        except (subprocess.TimeoutExpired, OSError) as error:
            raise SandboxUnavailableError(f"Docker did not answer: {error}") from error
        if probe.returncode != 0:
            raise SandboxUnavailableError(
                f"Docker cannot find the image {self._rules.image}: is the daemon "
                f"running, and is the image pulled (docker pull {self._rules.image})? "
                "A sandbox never pulls on its own and never falls back to the host."
            )

    def _refuse(self, refusal: Refusal) -> NoReturn:
        self._on_refusal(refusal)
        raise CommandRefusedError(refusal)


def _checked(rules: Rules) -> Rules:
    """Rules with resolved paths, or ValueError for one Docker cannot mount safely."""
    if not _IMAGE.match(rules.image):
        raise ValueError(f"{rules.image!r} is not an image reference")
    resolved = replace(
        rules,
        root=rules.root.resolve(),
        read=frozenset(path.resolve() for path in rules.read),
    )
    sockets = _docker_sockets()
    for path in (resolved.root, *resolved.read):
        text = str(path)
        if "," in text or "=" in text or not text.isprintable():
            raise ValueError(f"{text!r} cannot be mounted: no commas or equals signs")
        if any(path.is_relative_to(tree) for tree in _HOST_ONLY):
            raise ValueError(f"{text} is a host system tree and cannot be mounted")
        if path in resolved.read and (
            path.is_relative_to(WORKSPACE) or Path(WORKSPACE).is_relative_to(path)
        ):
            raise ValueError(f"{text} would cover the workspace at {WORKSPACE}")
        if any(socket.is_relative_to(path) for socket in sockets):
            raise ValueError(f"{text} contains the Docker socket and cannot be mounted")
    return resolved


def _docker_config_dir() -> Path:
    """Where the Docker CLI looks, as it does: an empty `DOCKER_CONFIG` is unset, and a
    `~` in a set one is not expanded."""
    value = os.environ.get("DOCKER_CONFIG")
    return Path(value) if value else Path.home() / ".docker"


def _docker_config() -> dict[str, object]:
    """The CLI's `config.json`: empty if there is none; a file that is there but
    cannot be read as a JSON object makes the sandbox unavailable, not unchecked."""
    path = _docker_config_dir() / "config.json"
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        raise SandboxUnavailableError(
            f"cannot read {path} to check it: {error}"
        ) from error
    if not isinstance(data, dict):
        raise SandboxUnavailableError(f"{path} is not a JSON object")
    return data


def _context_socket() -> Path | None:
    """The socket of the Docker context in use (`DOCKER_CONTEXT`, else the config's
    `currentContext`), which is where Colima, OrbStack and rootless Docker live. None
    for the default context. A context named but not found, or not a local socket, is
    an error: the socket could be anywhere, or the daemon on another machine."""
    name = os.environ.get("DOCKER_CONTEXT") or _docker_config().get("currentContext")
    if not name or name == "default":
        return None
    if not isinstance(name, str):
        raise SandboxUnavailableError("the Docker config's currentContext is not text")
    digest = hashlib.sha256(name.encode()).hexdigest()
    meta = _docker_config_dir() / "contexts" / "meta" / digest / "meta.json"
    try:
        host = json.loads(meta.read_text())["Endpoints"]["docker"]["Host"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SandboxUnavailableError(
            f"cannot find the endpoint of the Docker context {name!r} ({meta}): {error}"
        ) from error
    return _local_socket(host, f"the Docker context {name!r}")


def _local_socket(host: object, where: str) -> Path:
    """The path of a `unix://` endpoint. Any other daemon is remote, and a bind mount
    would name a path on the other machine: no sandbox."""
    if isinstance(host, str) and host.startswith("unix://"):
        return Path(host.removeprefix("unix://"))
    raise SandboxUnavailableError(
        f"{where} points at {host!r}, not a local unix socket: a remote daemon cannot "
        "see the workspace, so there is no sandbox on it"
    )


def _docker_sockets() -> list[Path]:
    """Where the Docker daemon's socket may be. A read-only mount does not stop a
    command from connecting to a socket, and the daemon is root on the host."""
    home = Path.home()
    found = [
        Path("/var/run/docker.sock"),
        Path("/run/docker.sock"),
        home / ".docker" / "run" / "docker.sock",
        home / ".docker" / "desktop" / "docker.sock",
    ]
    host = os.environ.get("DOCKER_HOST", "")
    if host:
        found.append(_local_socket(host, "DOCKER_HOST"))
    if (context := _context_socket()) is not None:
        found.append(context)
    # Both where it really is and where a link to it sits: a tree holding the link
    # is as unsafe as one holding the socket once the target is reachable too.
    return [
        form
        for path in found
        for form in (path.resolve(), path.parent.resolve() / path.name)
    ]


def _non_root_user() -> str:
    uid, gid = os.getuid(), os.getgid()
    if uid == 0 or gid == 0:
        raise SandboxUnavailableError(
            "loomcore is running as root, so the container's user would be root; "
            "run it as an ordinary user"
        )
    return f"{uid}:{gid}"


def _find_docker(docker: str) -> str:
    found = shutil.which(docker)
    if found is None:
        raise SandboxUnavailableError(f"{docker} is not installed or not on the path")
    return found


def _size(file: IO[bytes]) -> int:
    return os.fstat(file.fileno()).st_size


def _read_together(out: IO[bytes], err: IO[bytes], cap: int) -> tuple[str, str, bool]:
    """stdout and stderr, at most `cap` bytes between them; stdout comes first. The
    flag says that some output was dropped."""
    out.seek(0)
    err.seek(0)
    first = out.read(cap)
    second = err.read(max(cap - len(first), 0))
    cut = _size(out) > len(first) or _size(err) > len(second)
    return first.decode(errors="replace"), second.decode(errors="replace"), cut
