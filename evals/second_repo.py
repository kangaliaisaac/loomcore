"""A second repository's view of loomcore (spec section 6).

Run by `scripts/second-repo.sh` in a throwaway environment that holds only the built
wheel, so it can import nothing but the installed package. It imports only the names
at the top of `loomcore`, registers two tools of its own, runs a three-step task on a
scripted model with the real clocks, and reads the persisted log back.
"""

import tempfile
from decimal import Decimal
from pathlib import Path

from pydantic import JsonValue

import loomcore
from loomcore import (
    DATA_RULE,
    Config,
    Driver,
    Limits,
    ModelRequest,
    Ports,
    PromptParts,
    Settings,
    Task,
    build_prompt,
    finish,
    read_events,
    replay_events,
    system_ports,
    tool_kit,
)

MODEL = "claude-sonnet-5-5"


def multiply(a: int, b: int) -> str:
    """Multiply two whole numbers."""
    return str(a * b)


def add(a: int, b: int) -> str:
    """Add two whole numbers."""
    return str(a + b)


def reply(n: int, name: str, **args: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{n}",
        "model": MODEL,
        "role": "assistant",
        "content": [{"type": "tool_use", "id": f"t{n}", "name": name, "input": args}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100 * n, "output_tokens": 20},
    }


class Script:
    """The model port: three canned replies, one per call."""

    def __init__(self) -> None:
        self.replies = [
            reply(1, "multiply", a=6, b=7),
            reply(2, "add", a=42, b=8),
            reply(3, finish.NAME, result="50"),
        ]

    def __call__(self, request: ModelRequest) -> dict[str, JsonValue]:
        return self.replies.pop(0)


def main() -> None:
    assert "site-packages" in (loomcore.__file__ or ""), "not the installed wheel"
    root = Path(tempfile.mkdtemp(prefix="second-repo-"))
    kit = tool_kit(root)
    kit.registry.register(multiply, read_only=True)
    kit.registry.register(add, read_only=True)
    prompt = build_prompt(
        PromptParts(
            role="You do arithmetic with the tools you are given.",
            rules=(DATA_RULE,),
            tools=tuple(kit.registry.schemas()),
        )
    )
    schemas = tuple(schema.to_json() for schema in kit.registry.schemas())
    config = Config(MODEL, prompt.text, (*schemas, finish.SCHEMA))
    ports = Ports(model=Script(), **kit.tool_ports(), **system_ports())
    session = root / "session.jsonl"
    driver = Driver(config, ports, Settings(session, root / "spend.jsonl"))
    result = driver.run(
        Task(
            "What is 6 * 7, plus 8?", Limits(run_cap=Decimal(1), month_cap=Decimal(10))
        )
    )
    assert (result.status, result.result) == ("finished", "50"), result

    events = read_events(session).events  # the log is on disk, not in the driver
    state = replay_events(events)
    assert state.finished == "finished" and state.turns == 3, state
    assert [e.kind for e in events].count("tool_completed") == 3
    print(f"ok: {len(events)} events persisted and replayed to {state.finished}")


if __name__ == "__main__":
    main()
