"""f17: the system-prompt builder is deterministic and prefix-stable."""

from hypothesis import given
from hypothesis import strategies as st

from loomcore.prompt import STABLE, Block, PromptParts, SystemPrompt, build
from loomcore.tools import ToolSchema

READ = ToolSchema("read_file", "Read a file in\n the workspace.", {})
WRITE = ToolSchema("write_note", "Append to the notes file.", {})
PARTS = PromptParts(
    role="You are a careful assistant.",
    rules=("Answer briefly.", "Never guess a path."),
    tools=(WRITE, READ),
    environment="Today is a Monday.",
)
SNAPSHOT = """You are a careful assistant.

Rules:
- Answer briefly.
- Never guess a path.

Tools:
- read_file: Read a file in the workspace.
- write_note: Append to the notes file.

Environment:
Today is a Monday."""


def test_the_same_parts_give_the_same_bytes() -> None:
    assert build(PARTS).text == SNAPSHOT
    assert build(PARTS) == build(PARTS)


def test_the_order_tools_were_registered_in_does_not_change_the_bytes() -> None:
    swapped = PromptParts(PARTS.role, PARTS.rules, (READ, WRITE), PARTS.environment)
    assert build(swapped).text == build(PARTS).text


def test_the_cache_breakpoint_is_the_last_stable_block() -> None:
    prompt = build(PARTS)
    assert [block.name for block in prompt.blocks] == [
        "role",
        "rules",
        "tools",
        "environment",
    ]
    assert prompt.cache_breakpoint == 2
    assert prompt.stable_text == SNAPSHOT.split("\n\nEnvironment:", maxsplit=1)[0]


def test_an_empty_part_leaves_out_its_block_and_moves_the_breakpoint() -> None:
    prompt = build(PromptParts(role="Role.", environment="Env."))
    assert [block.name for block in prompt.blocks] == ["role", "environment"]
    assert prompt.cache_breakpoint == 0
    assert build(PromptParts(role="", environment="Env.")).cache_breakpoint is None


def test_changing_the_environment_leaves_the_whole_stable_prefix_alone() -> None:
    other = PromptParts(PARTS.role, PARTS.rules, PARTS.tools, "Today is a Friday.")
    assert build(other).stable_text == build(PARTS).stable_text
    assert build(other).text != build(PARTS).text


TEXT = st.text(alphabet="abc xyz.\n", min_size=1, max_size=12).filter(
    lambda s: s.strip() != ""
)
PARTS_STRATEGY = st.builds(
    PromptParts,
    role=TEXT,
    rules=st.lists(TEXT, max_size=3).map(tuple),
    tools=st.lists(
        st.builds(ToolSchema, name=TEXT, description=TEXT, input_schema=st.just({})),
        max_size=3,
        unique_by=lambda tool: tool.name,
    ).map(tuple),
    environment=TEXT,
)
ORDER = ("role", "rules", "tools", "environment")


def changed(parts: PromptParts, part: str, text: str) -> PromptParts:
    match part:
        case "role":
            return PromptParts(text, parts.rules, parts.tools, parts.environment)
        case "rules":
            return PromptParts(
                parts.role, (*parts.rules, text), parts.tools, parts.environment
            )
        case "tools":
            extra = ToolSchema("zz_" + text.strip(), "x", {})
            return PromptParts(
                parts.role, parts.rules, (*parts.tools, extra), parts.environment
            )
        case _:
            return PromptParts(parts.role, parts.rules, parts.tools, text)


@given(PARTS_STRATEGY, st.sampled_from(ORDER), TEXT)
def test_a_changed_part_changes_only_the_blocks_from_itself_onwards(
    parts: PromptParts, part: str, text: str
) -> None:
    before, after = build(parts), build(changed(parts, part, text))
    cut = ORDER.index(part)
    untouched = [b for b in before.blocks if ORDER.index(b.name) < cut]
    again = [b for b in after.blocks if ORDER.index(b.name) < cut]
    assert untouched == again  # nothing before the changed part moved


@given(PARTS_STRATEGY)
def test_the_stable_prefix_never_holds_the_environment(parts: PromptParts) -> None:
    prompt: SystemPrompt = build(parts)
    assert all(b.name in STABLE for b in prompt.blocks if b.name != "environment")
    assert "Environment:" not in prompt.stable_text
    assert Block("environment", "x").name not in STABLE
