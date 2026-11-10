"""What the Docker command-line client is set up to do, read from its own config and
environment (`dockerenv.py`): the socket it talks to, and anything it would add to a
container. The sandbox checks these once, when it is built.
"""

import hashlib
import json
import os
from pathlib import Path


class SandboxUnavailableError(Exception):
    """Docker or the image is not there. The sandbox is not built, and nothing runs
    on the host instead."""


def check_no_proxies() -> None:
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


def docker_sockets() -> list[Path]:
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
    try:
        context = _context_socket()
    except SandboxUnavailableError:
        if not host:
            raise
        context = None  # DOCKER_HOST decides which daemon is used; this one is not
    if host:
        found.append(_local_socket(host, "DOCKER_HOST"))
    if context is not None:  # still refused as a tree: that daemon may be running
        found.append(context)
    # Both where it really is and where a link to it sits: a tree holding the link
    # is as unsafe as one holding the socket once the target is reachable too.
    return [
        form
        for path in found
        for form in (path.resolve(), path.parent.resolve() / path.name)
    ]
