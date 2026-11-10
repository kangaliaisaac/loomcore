"""f12: deny by default; each command in a container with only what the rules gave.

The rule and argument tests need no Docker. The rest start real containers from a
small local image, and are skipped without Docker (but fail in CI, so the sandbox is
never silently untested there).
"""

import itertools
import os
import shutil
import stat
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from loomcore.sandbox import (
    LABEL,
    CommandRefusedError,
    DockerSandbox,
    Launch,
    Permit,
    ProcessResult,
    Refusal,
    Request,
    ResourceLimits,
    Rules,
    SandboxUnavailableError,
    docker_args,
    env_value_problem,
    evaluate,
    write_env_file,
)

# Ubuntu 24.04 LTS, pinned by digest: the tag can move, the digest cannot. An image is
# the owner's choice (Rules.image); this is only the one the tests run in.
IMAGE = os.environ.get(
    "LOOMCORE_TEST_IMAGE",
    "ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852",
)
SH, CAT, ENV, ECHO = "/bin/sh", "/bin/cat", "/usr/bin/env", "/bin/echo"
BASH, GETENT = "/bin/bash", "/usr/bin/getent"
SLEEP, PWD, TOUCH, YES = "/bin/sleep", "/bin/pwd", "/bin/touch", "/usr/bin/yes"
ALL = frozenset({SH, CAT, ENV, ECHO, SLEEP, PWD, TOUCH, YES, BASH, GETENT})


def docker_has_the_image() -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    probe = subprocess.run(
        [docker, "image", "inspect", IMAGE], capture_output=True, check=False
    )
    return probe.returncode == 0


@pytest.fixture(scope="session")
def docker() -> None:
    if docker_has_the_image():
        return
    message = f"Docker and the image {IMAGE} are needed (docker pull {IMAGE})"
    if os.environ.get("CI"):
        pytest.fail(message)  # CI must run these, not skip them
    pytest.skip(message)


needs_docker = pytest.mark.usefixtures("docker")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "work").mkdir()
    return (tmp_path / "work").resolve()


def rules_for(root: Path, **changes: object) -> Rules:
    base = {"root": root, "image": IMAGE, "commands": ALL}
    return Rules(**{**base, **changes})  # type: ignore[arg-type]


def ask(
    command: tuple[str, ...], cwd: Path, env: frozenset[str] = frozenset()
) -> Request:
    return Request(command, cwd, env)


# --- evaluate: pure, deny by default ------------------------------------------------


def test_everything_is_refused_by_default(root: Path) -> None:
    nothing = Rules(root=root, image=IMAGE, commands=frozenset())
    verdict = evaluate(nothing, ask((SH,), root))
    assert isinstance(verdict, Refusal)
    assert verdict.action == "exec"
    assert "not an allowed command" in verdict.reason


def test_an_allowed_command_in_the_workspace_is_permitted(root: Path) -> None:
    assert evaluate(rules_for(root), ask((SH, "-c", "true"), root)) == Permit()
    assert evaluate(rules_for(root), ask((SH,), root / "sub")) == Permit()


@pytest.mark.parametrize("field", list(vars(ResourceLimits())))
@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_a_limit_that_docker_reads_as_unlimited_is_refused(
    root: Path, field: str, bad: float
) -> None:
    limits = replace(ResourceLimits(), **{field: bad})  # type: ignore[arg-type]
    request = Request((SH,), root, frozenset())
    launch = Launch("docker", "n", Path("/e"), "1:1")
    with pytest.raises(ValueError, match=field):
        docker_args(rules_for(root), request, limits, launch)


@pytest.mark.parametrize(
    ("command", "why"),
    [
        ((), "no command"),
        (("sh",), "absolute path"),
        (("./sh",), "absolute path"),
        (("/bin/rm", "-rf", "/"), "not an allowed command"),
        (("/bin/sh/../rm",), "not an allowed command"),  # compared as written
    ],
)
def test_a_command_that_is_not_an_allowed_absolute_path_is_refused(
    root: Path, command: tuple[str, ...], why: str
) -> None:
    verdict = evaluate(rules_for(root), ask(command, root))
    assert isinstance(verdict, Refusal)
    assert why in verdict.reason


def test_a_working_directory_outside_the_workspace_is_refused(root: Path) -> None:
    for cwd in (root.parent, Path("/"), root.parent / (root.name + "2"), Path("rel")):
        verdict = evaluate(rules_for(root), ask((SH,), cwd))
        assert isinstance(verdict, Refusal), cwd
        assert "outside the workspace" in verdict.reason


def test_an_environment_variable_the_rules_do_not_allow_is_refused(root: Path) -> None:
    rules = rules_for(root, env=frozenset({"LANG"}))
    assert evaluate(rules, ask((SH,), root, frozenset({"LANG"}))) == Permit()
    verdict = evaluate(rules, ask((SH,), root, frozenset({"LANG", "API_KEY"})))
    assert isinstance(verdict, Refusal)
    assert "API_KEY" in verdict.reason
    assert "LANG" not in verdict.reason


def test_a_variable_name_that_is_not_a_name_is_refused(root: Path) -> None:
    rules = rules_for(root, env=frozenset({"A=b", "-x"}))
    for bad in ("A=b", "-x"):
        verdict = evaluate(rules, ask((SH,), root, frozenset({bad})))
        assert isinstance(verdict, Refusal)
        assert "not a variable name" in verdict.reason


NEAR_MISSES = st.sampled_from(
    ["/bin/sh", "/bin/sh2", "/bin/s", "/bin/sh/", "/bin/sh/../rm", "/bin//sh", "sh", ""]
)


@given(
    program=st.one_of(st.text(max_size=12), NEAR_MISSES),
    args=st.lists(st.text(max_size=8), max_size=3),
)
def test_no_command_outside_the_allow_list_is_ever_permitted(
    program: str, args: list[str]
) -> None:
    rules = Rules(root=Path("/work"), image=IMAGE, commands=frozenset({"/bin/sh"}))
    verdict = evaluate(rules, ask((program, *args), Path("/work")))
    clean = not any(c in a for a in args for c in "\x00\ud800")
    assert (verdict == Permit()) == (program == "/bin/sh" and clean)


# --- docker_args: the isolation is in the command line, and fixed ---------------------


def args_for(
    root: Path, *, cwd: Path | None = None, read: frozenset[Path] = frozenset()
) -> list[str]:
    rules = rules_for(root, read=read)
    request = Request((SH, "-c", "echo hi"), cwd or root, frozenset())
    launch = Launch("docker", "loomcore-test", Path("/private/env"), "1000:1000")
    return docker_args(rules, request, ResourceLimits(), launch)


def pairs(args: list[str]) -> set[tuple[str, str]]:
    return set(itertools.pairwise(args))


def test_the_command_line_isolates_the_container(root: Path) -> None:
    args = args_for(root)
    assert {
        ("--network", "none"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--pull", "never"),
        ("--user", "1000:1000"),
        ("--ipc", "none"),
        ("--workdir", "/workspace"),
        ("--env-file", "/private/env"),
        ("--label", LABEL),
    } <= pairs(args)
    assert "--read-only" in args
    assert "--rm" in args
    assert "--init" in args
    assert args[args.index(IMAGE) + 1 :] == [SH, "-c", "echo hi"]


def test_nothing_that_widens_the_container_is_on_the_command_line(root: Path) -> None:
    args = args_for(root)
    forbidden = {"--privileged", "--cap-add", "--pid", "--uts", "--device"}
    forbidden |= {"--volume", "-v", "--volumes-from", "--net", "--userns"}
    assert not forbidden & set(args)
    assert ("--network", "host") not in pairs(args)
    assert ("--ipc", "host") not in pairs(args)
    assert not any("docker.sock" in a or "unconfined" in a for a in args)
    assert ("--user", "0") not in pairs(args)
    assert ("--user", "root") not in pairs(args)


def test_the_workspace_is_the_only_writable_mount(root: Path, tmp_path: Path) -> None:
    extra = (tmp_path / "libs").resolve()
    args = args_for(root, read=frozenset({extra}))
    mounts = [args[i + 1] for i, a in enumerate(args) if a == "--mount"]
    writable = [m for m in mounts if not m.endswith(",readonly")]
    assert writable == [f"type=bind,src={root},dst=/workspace"]
    assert mounts[1:] == [f"type=bind,src={extra},dst={extra},readonly"]


def test_a_working_directory_below_the_root_is_mapped_into_the_workspace(
    root: Path,
) -> None:
    args = args_for(root, cwd=root / "a" / "b")
    assert ("--workdir", "/workspace/a/b") in pairs(args)


def test_every_resource_limit_is_on_the_command_line(root: Path) -> None:
    limits = ResourceLimits(memory_bytes=1000, pids=7, cpus=0.5, cpu_seconds=3)
    request = Request((SH,), root, frozenset())
    launch = Launch("docker", "n", Path("/e"), "1:1")
    args = docker_args(rules_for(root), request, limits, launch)
    assert {
        ("--memory", "1000"),
        ("--memory-swap", "1000"),  # no swap to hide behind
        ("--pids-limit", "7"),
        ("--cpus", "0.5"),
        ("--ulimit", "cpu=3:3"),
        ("--ulimit", f"fsize={limits.max_file_bytes}:{limits.max_file_bytes}"),
        ("--ulimit", f"nofile={limits.max_open_files}:{limits.max_open_files}"),
        ("--ulimit", "core=0:0"),
    } <= pairs(args)


@pytest.mark.parametrize(
    "user", ["0:0", "0:1000", "1000:0", "root", "1000", "", "1000:1000:1", "-1:5"]
)
def test_a_user_that_is_root_or_malformed_is_refused(root: Path, user: str) -> None:
    request = Request((SH,), root, frozenset())
    launch = Launch("docker", "n", Path("/e"), user)
    with pytest.raises(ValueError, match="non-root"):
        docker_args(rules_for(root), request, ResourceLimits(), launch)


@pytest.mark.parametrize("which", ["getuid", "getgid"])
def test_a_sandbox_is_not_built_when_loomcore_runs_as_root(
    root: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    monkeypatch.setattr(os, which, lambda: 0)
    with pytest.raises(SandboxUnavailableError, match="root"):
        DockerSandbox(rules_for(root), on_refusal=lambda _: None, docker="true")


def test_an_image_that_could_be_taken_for_an_option_is_refused(root: Path) -> None:
    for bad in ("--privileged", "-v", "alpine --privileged", "", "a b"):
        with pytest.raises(ValueError, match="image reference"):
            DockerSandbox(
                rules_for(root, image=bad), on_refusal=lambda _: None, docker="true"
            )


def test_a_path_that_cannot_be_mounted_safely_is_refused(tmp_path: Path) -> None:
    for name in ("a,b", "a=b"):
        (tmp_path / name).mkdir()
        with pytest.raises(ValueError, match="cannot be mounted"):
            DockerSandbox(
                rules_for(tmp_path / name), on_refusal=lambda _: None, docker="true"
            )


def test_a_tree_that_contains_the_docker_socket_cannot_be_mounted(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only mount does not stop a command connecting to a socket, and the
    daemon is root on the host."""
    sockets = tmp_path / "run"
    sockets.mkdir()
    monkeypatch.setenv("DOCKER_HOST", f"unix://{sockets}/d.sock")
    for tree in (sockets, tmp_path, Path("/"), Path.home(), Path("/var/run")):
        with pytest.raises(ValueError, match="Docker socket"):
            DockerSandbox(
                rules_for(root, read=frozenset({tree})),
                on_refusal=lambda _: None,
                docker="true",
            )
    with pytest.raises(ValueError, match="Docker socket"):  # nor as the workspace
        DockerSandbox(rules_for(Path.home()), on_refusal=lambda _: None, docker="true")


def test_values_the_environment_file_cannot_hold_are_named() -> None:
    assert env_value_problem("A", "fine value") is None
    for bad in ("a\nb", "a\rb", "a\x00b", "\udcff", "x" * 70_000):
        assert env_value_problem("A", bad) is not None, repr(bad)


def test_the_environment_file_is_private_and_holds_the_variables(
    tmp_path: Path,
) -> None:
    path = write_env_file(tmp_path, {"A": "1", "B": "two words"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_text() == "A=1\nB=two words\n"


# --- the sandbox is not built without Docker ------------------------------------------


def test_without_the_docker_executable_there_is_no_sandbox(root: Path) -> None:
    with pytest.raises(SandboxUnavailableError, match="not installed"):
        DockerSandbox(
            rules_for(root), on_refusal=lambda _: None, docker="/nonexistent/docker"
        )


def test_a_docker_proxy_config_means_no_sandbox(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "dockercfg"
    config.mkdir()
    (config / "config.json").write_text('{"proxies": {"default": {"httpProxy": "x"}}}')
    monkeypatch.setenv("DOCKER_CONFIG", str(config))
    with pytest.raises(SandboxUnavailableError, match="proxies"):
        DockerSandbox(rules_for(root), on_refusal=lambda _: None, docker="true")


def test_an_unreadable_docker_config_means_no_sandbox(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "dockercfg"
    config.mkdir()
    (config / "config.json").write_text("{not json")
    monkeypatch.setenv("DOCKER_CONFIG", str(config))
    with pytest.raises(SandboxUnavailableError, match="cannot read"):
        DockerSandbox(rules_for(root), on_refusal=lambda _: None, docker="true")


@needs_docker
def test_without_the_image_there_is_no_sandbox(root: Path) -> None:
    with pytest.raises(SandboxUnavailableError, match="docker pull"):
        DockerSandbox(
            rules_for(root, image="loomcore-no-such-image:0"),
            on_refusal=lambda _: None,
        )


# --- real containers ------------------------------------------------------------------


class Watched:
    def __init__(self, rules: Rules, environ: dict[str, str] | None = None) -> None:
        self.refusals: list[Refusal] = []
        self.sandbox = DockerSandbox(
            rules, environ=environ or {}, on_refusal=self.refusals.append
        )

    def run(
        self,
        *command: str,
        cwd: Path,
        env: frozenset[str] = frozenset(),
        limits: ResourceLimits = ResourceLimits(timeout=60),  # noqa: B008
    ) -> ProcessResult:
        return self.sandbox.run(command, cwd=cwd, env_allow=env, limits=limits)


def leftover_containers() -> str:
    done = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label={LABEL}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.strip()


@needs_docker
def test_a_secret_in_the_parent_is_not_passed_to_the_command(root: Path) -> None:
    parent = {"SECRET_TOKEN": "hunter2", "VISIBLE": "yes"}
    watched = Watched(rules_for(root, env=frozenset({"VISIBLE"})), parent)
    result = watched.run(ENV, cwd=root, env=frozenset({"VISIBLE"}))
    names = {line.split("=")[0] for line in result.stdout.split()}
    assert names == {"VISIBLE", "HOME", "HOSTNAME", "PATH"}  # + only the image's own
    assert "hunter2" not in result.stdout + result.stderr


@needs_docker
def test_a_secret_in_the_real_process_environment_is_not_passed_either(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOOMCORE_TEST_SECRET", "hunter2")
    sandbox = DockerSandbox(rules_for(root), on_refusal=lambda _: None)  # os.environ
    result = sandbox.run(
        [SH, "-c", "env; cat /proc/*/environ"], cwd=root, env_allow=set()
    )
    assert "hunter2" not in result.stdout + result.stderr


@needs_docker
def test_the_command_cannot_see_the_hosts_files_or_processes(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOOMCORE_TEST_SECRET", "hunter2")
    outside = tmp_path / "outside.txt"
    outside.write_text("host secret")
    watched = Watched(rules_for(root))
    read = watched.run(CAT, str(outside), cwd=root)
    assert read.returncode != 0
    assert "host secret" not in read.stdout
    processes = watched.run(SH, "-c", "ls /proc | grep -c '^[0-9]'", cwd=root)
    assert int(processes.stdout) < 10  # its own pid namespace, not the host's
    parent = watched.run(SH, "-c", f"cat /proc/{os.getpid()}/environ", cwd=root)
    assert parent.returncode != 0  # the loomcore process is not there to read
    assert "hunter2" not in parent.stdout


@needs_docker
def test_the_command_can_write_the_workspace_and_nothing_else(root: Path) -> None:
    watched = Watched(rules_for(root))
    inside = watched.run(SH, "-c", "echo made > /workspace/new.txt", cwd=root)
    assert inside.returncode == 0
    assert (root / "new.txt").read_text() == "made\n"
    assert (root / "new.txt").stat().st_uid == os.getuid()  # not root's
    # /var/tmp is world-writable in the image, so only a read-only root filesystem
    # (not file permissions) stops a write there; /etc and /usr would fail for a
    # non-root user either way.
    for outside in (
        "/var/tmp/pwn",
        "/etc/pwn",
        "/usr/pwn",
        "/pwn",
        "/workspace/../pwn",
    ):
        result = watched.run(SH, "-c", f"echo x > {outside}", cwd=root)
        assert result.returncode != 0, outside
    assert not (root.parent / "pwn").exists()
    read_only = watched.run(SH, "-c", "echo x > /var/tmp/pwn", cwd=root)
    assert "Read-only file system" in read_only.stderr


@needs_docker
def test_the_scratch_directory_cannot_run_programs(root: Path) -> None:
    watched = Watched(rules_for(root))
    script = "cp /bin/ls /tmp/b && chmod +x /tmp/b && /tmp/b"
    result = watched.run(SH, "-c", script, cwd=root)
    assert result.returncode != 0
    assert "Permission denied" in result.stderr


@needs_docker
def test_the_command_is_not_root_and_has_no_capabilities(root: Path) -> None:
    watched = Watched(rules_for(root))
    ids = watched.run(SH, "-c", "id -u", cwd=root)
    assert ids.stdout.strip() == str(os.getuid()) != "0"
    status = watched.run(
        SH, "-c", "grep -E 'Cap|NoNewPrivs' /proc/self/status", cwd=root
    )
    for field in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        assert f"{field}:\t0000000000000000" in status.stdout, field  # none at all
    assert "NoNewPrivs:\t1" in status.stdout


@needs_docker
def test_the_command_has_no_network(root: Path) -> None:
    """With networking on there is a default route and names resolve; here neither.

    (The routing table is the reliable sign: the container's /sys/class/net lists
    kernel tunnel devices even with no network, and a host-side server proves
    nothing on Linux, where a bridged container cannot reach the host's loopback.)
    """
    watched = Watched(rules_for(root))
    routes = watched.run(SH, "-c", "cat /proc/net/route", cwd=root)
    assert len(routes.stdout.strip().splitlines()) == 1  # the header and no route
    resolved = watched.run(GETENT, "hosts", "example.com", cwd=root)
    assert resolved.returncode != 0  # no name resolution
    for address in ("1.1.1.1/80", "127.0.0.1/9", "example.com/80"):
        connect = f"exec 3<>/dev/tcp/{address}"
        reached = watched.run(BASH, "-c", connect, cwd=root)
        assert reached.returncode != 0, address


@needs_docker
def test_there_is_no_shared_memory_to_write_to(root: Path) -> None:
    watched = Watched(rules_for(root))
    result = watched.run(SH, "-c", "echo x > /dev/shm/x", cwd=root)
    assert result.returncode != 0  # --ipc none: no /dev/shm at all


@needs_docker
def test_a_refusal_is_logged_and_nothing_runs(root: Path) -> None:
    watched = Watched(rules_for(root, commands=frozenset({ECHO})))
    with pytest.raises(CommandRefusedError) as raised:
        watched.run(TOUCH, "/workspace/ran", cwd=root)
    assert watched.refusals == [raised.value.refusal]
    assert watched.refusals[0].action == "exec"
    assert not (root / "ran").exists()
    assert leftover_containers() == ""


@needs_docker
def test_every_kind_of_refusal_is_logged(root: Path) -> None:
    watched = Watched(rules_for(root))
    attempts: list[tuple[tuple[str, ...], Path, frozenset[str]]] = [
        ((), root, frozenset()),
        (("echo",), root, frozenset()),
        ((ECHO,), root.parent, frozenset()),
        ((ECHO,), root, frozenset({"AWS_SECRET"})),
    ]
    for command, cwd, env in attempts:
        with pytest.raises(CommandRefusedError):
            watched.run(*command, cwd=cwd, env=env)
    assert len(watched.refusals) == 4


@needs_docker
def test_the_command_is_not_run_by_a_shell(root: Path) -> None:
    watched = Watched(rules_for(root))
    result = watched.run(ECHO, "a; touch pwned && $(id) `id` *", cwd=root)
    assert result.stdout == "a; touch pwned && $(id) `id` *\n"
    assert not (root / "pwned").exists()


@needs_docker
def test_the_command_runs_in_the_given_directory(root: Path) -> None:
    (root / "sub").mkdir()
    watched = Watched(rules_for(root))
    assert watched.run(PWD, cwd=root / "sub").stdout.strip() == "/workspace/sub"
    assert watched.run(PWD, cwd=Path("sub")).stdout.strip() == "/workspace/sub"
    with pytest.raises(CommandRefusedError):  # relative to the workspace, so no escape
        watched.run(PWD, cwd=Path(".."))


@needs_docker
@pytest.mark.parametrize("bad", ["a\x00b", "\ud800"])
def test_an_unusable_working_directory_or_argument_is_a_logged_refusal(
    root: Path, bad: str
) -> None:
    watched = Watched(rules_for(root))
    with pytest.raises(CommandRefusedError):
        watched.run(ECHO, cwd=Path(bad))
    with pytest.raises(CommandRefusedError):
        watched.run(ECHO, bad, cwd=root)
    assert len(watched.refusals) == 2


@needs_docker
@pytest.mark.parametrize("bad", ["line one\nline two", "ends with\r", "\udcff"])
def test_a_value_the_environment_file_cannot_hold_is_refused(
    root: Path, bad: str
) -> None:
    rules = rules_for(root, env=frozenset({"ODD"}))
    watched = Watched(rules, {"ODD": bad})
    with pytest.raises(CommandRefusedError, match="cannot be passed"):
        watched.run(ENV, cwd=root, env=frozenset({"ODD"}))
    assert len(watched.refusals) == 1


@needs_docker
def test_a_missing_directory_or_program_is_a_result_not_a_crash(root: Path) -> None:
    (root / "file").write_text("x")
    watched = Watched(rules_for(root, commands=ALL | {"/nope"}))
    for cwd in (root / "missing", root / "file"):
        result = watched.run(ECHO, cwd=cwd)
        assert (result.returncode, "not a directory" in result.stderr) == (127, True)
    assert watched.run("/nope", cwd=root).returncode == 127
    assert watched.refusals == []  # a typo is not a rule violation


@needs_docker
def test_a_timeout_kills_the_container_and_leaves_none_behind(root: Path) -> None:
    watched = Watched(rules_for(root))
    started = time.monotonic()
    result = watched.run(
        SH, "-c", "sleep 60 & wait", cwd=root, limits=ResourceLimits(timeout=2)
    )
    assert result.timed_out
    assert result.returncode == 137
    assert time.monotonic() - started < 30
    assert leftover_containers() == ""  # its background child went with it


@needs_docker
def test_a_timeout_that_arrives_before_the_container_exists_still_ends_the_run(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`docker kill` misses a container Docker has not created yet. The removals
    below are made to miss, as they would on a slow daemon; the run must keep trying
    and end, not wait for the command to finish by itself (60 s)."""
    real = DockerSandbox._remove
    misses = {"left": 6}

    def flaky(self: DockerSandbox, name: str) -> None:
        if misses["left"] > 0:
            misses["left"] -= 1
            return
        real(self, name)

    monkeypatch.setattr(DockerSandbox, "_remove", flaky)
    watched = Watched(rules_for(root))
    started = time.monotonic()
    result = watched.run(SLEEP, "60", cwd=root, limits=ResourceLimits(timeout=0.05))
    assert result.timed_out
    assert time.monotonic() - started < 30
    assert leftover_containers() == ""


@needs_docker
def test_output_that_exceeds_the_cap_is_flagged_even_if_the_command_exits_first(
    root: Path,
) -> None:
    watched = Watched(rules_for(root))
    result = watched.run(
        SH, "-c", "head -c 20000 /dev/zero | tr '\\0' a", cwd=root,
        limits=ResourceLimits(timeout=60, max_output_bytes=5000),
    )  # fmt: skip
    assert result.returncode == 0  # it finished by itself
    assert result.output_truncated
    assert len(result.stdout) == 5000


@needs_docker
def test_the_output_cap_covers_stdout_and_stderr_together(root: Path) -> None:
    watched = Watched(rules_for(root))
    script = (
        "head -c 4000 /dev/zero | tr '\\0' a; head -c 4000 /dev/zero | tr '\\0' b >&2"
    )
    result = watched.run(
        SH, "-c", script, cwd=root,
        limits=ResourceLimits(timeout=60, max_output_bytes=5000),
    )  # fmt: skip
    assert len(result.stdout) + len(result.stderr) == 5000
    assert (len(result.stdout), len(result.stderr)) == (4000, 1000)
    assert result.output_truncated


@needs_docker
def test_output_within_the_cap_is_not_flagged(root: Path) -> None:
    watched = Watched(rules_for(root))
    result = watched.run(ECHO, "hello", cwd=root)
    assert (result.stdout, result.output_truncated) == ("hello\n", False)


@needs_docker
def test_the_limits_reach_the_containers_cgroup(root: Path) -> None:
    """A command that is killed proves little: with no limit the VM's own out-of-memory
    killer would end it too. The values the kernel was given are the proof."""
    watched = Watched(rules_for(root))
    limits = ResourceLimits(timeout=60, memory_bytes=48 * 1024 * 1024, pids=21)
    script = (
        "for f in memory.max memory.swap.max pids.max; do "
        'echo "$f=$(cat /sys/fs/cgroup/$f)"; done'
    )
    result = watched.run(SH, "-c", script, cwd=root, limits=limits)
    seen = dict(line.split("=") for line in result.stdout.split())
    assert seen == {
        "memory.max": str(48 * 1024 * 1024),
        "memory.swap.max": "0",  # swap equal to memory means none to hide behind
        "pids.max": "21",
    }


@needs_docker
def test_the_memory_limit_kills_a_command_that_eats_memory(root: Path) -> None:
    watched = Watched(rules_for(root))
    result = watched.run(
        SH,
        "-c",
        "a=xxxxxxxx; while :; do a=$a$a; done",
        cwd=root,
        limits=ResourceLimits(timeout=60, memory_bytes=32 * 1024 * 1024),
    )
    assert not result.timed_out
    assert result.returncode == 137


@needs_docker
def test_the_process_limit_stops_a_fork_bomb(root: Path) -> None:
    watched = Watched(rules_for(root))
    result = watched.run(
        SH,
        "-c",
        "for i in $(seq 1 100); do sleep 2 & done; wait",
        cwd=root,
        limits=ResourceLimits(timeout=60, pids=16),
    )
    assert "fork" in result.stderr.lower()  # dash: "Cannot fork"
    assert not result.timed_out


@needs_docker
def test_the_cpu_limit_stops_a_busy_loop(root: Path) -> None:
    watched = Watched(rules_for(root))
    started = time.monotonic()
    result = watched.run(
        SH, "-c", "while :; do :; done", cwd=root,
        limits=ResourceLimits(timeout=60, cpu_seconds=1),
    )  # fmt: skip
    assert not result.timed_out
    assert result.returncode != 0
    assert time.monotonic() - started < 30


@needs_docker
def test_the_file_size_limit_caps_what_a_command_writes(root: Path) -> None:
    watched = Watched(rules_for(root))
    watched.run(
        SH, "-c", "yes > /workspace/big", cwd=root,
        limits=ResourceLimits(timeout=60, max_file_bytes=1000),
    )  # fmt: skip
    assert 0 < (root / "big").stat().st_size <= 1000


@needs_docker
def test_runaway_output_is_cut_and_the_container_killed(root: Path) -> None:
    watched = Watched(rules_for(root))
    result = watched.run(
        YES, cwd=root, limits=ResourceLimits(timeout=60, max_output_bytes=10_000)
    )
    assert result.output_truncated
    assert not result.timed_out
    assert len(result.stdout) <= 10_000
    assert leftover_containers() == ""


@needs_docker
def test_the_environment_file_does_not_outlive_the_run(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(scratch))
    watched = Watched(rules_for(root, env=frozenset({"TOKEN"})), {"TOKEN": "hunter2"})
    watched.run(ENV, cwd=root, env=frozenset({"TOKEN"}))
    assert list(scratch.iterdir()) == []
