"""A hello-agent: two tools, a three-step task, and a run capped at five cents."""

from decimal import Decimal
from pathlib import Path

from loomcore import (
    DATA_RULE,
    Config,
    Driver,
    Limits,
    Ports,
    PromptParts,
    Settings,
    Task,
    build_prompt,
    estimate_request,
    estimate_tokens,
    finish,
    system_ports,
    tool_kit,
)
from loomcore.anthropic import model


def multiply(a: int, b: int) -> str:
    """Multiply two whole numbers."""
    return str(a * b)


def add(a: int, b: int) -> str:
    """Add two whole numbers."""
    return str(a + b)


Path("work").mkdir(exist_ok=True)
kit = tool_kit(Path("work"))
kit.registry.register(multiply, read_only=True)
kit.registry.register(add, read_only=True)
schemas = kit.registry.schemas()
prompt = build_prompt(
    PromptParts(
        role="You do arithmetic with your tools.",
        rules=(
            "When you have the answer, call finish with the number alone.",
            DATA_RULE,
        ),
        tools=tuple(schemas),
    )
)
tools = (*(schema.to_json() for schema in schemas), finish.SCHEMA)
config = Config(
    "claude-haiku-5-5", prompt.text, tools, cache_prefix=len(prompt.stable_text)
)
counters = system_ports(count_tokens=estimate_tokens, count_request=estimate_request)
ports = Ports(model=model(), **kit.tool_ports(), **counters)
driver = Driver(config, ports, Settings(Path("session.jsonl"), Path("spend.jsonl")))
limits = Limits(run_cap=Decimal("0.05"), month_cap=Decimal("1"))
result = driver.run(Task("What is 6 * 7, plus 8?", limits))
print(result.status, result.limit or result.result)
