"""f17: the system-prompt builder is deterministic and prefix-stable."""

from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from loomcore.prompt import SEPARATOR, PromptParts, SystemPrompt, build
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


TEXT = st.text(alphabet="abc xyzEnv:-.\n", min_size=1, max_size=12).filter(
    lambda s: s.strip() != ""
)
PARTS_STRATEGY = st.builds(
    PromptParts,
    role=TEXT,
    rules=st.lists(TEXT, max_size=3).map(tuple),
    tools=st.lists(
        st.builds(ToolSchema, name=TEXT, description=TEXT, input_schema=st.just({})),
        max_size=3,
        unique_by=lambda tool: " ".join(tool.name.split()),
    ).map(tuple),
    environment=TEXT,
)
ORDER = ("role", "rules", "tools", "environment")


def changed(parts: PromptParts, part: str, text: str) -> PromptParts:
    match part:
        case "role":
            return replace(parts, role=text)
        case "rules":
            return replace(parts, rules=(*parts.rules, text))
        case "tools":
            extra = ToolSchema("zz_" + text.strip(), "x", {})
            return replace(parts, tools=(*parts.tools, extra))
        case _:
            return replace(parts, environment=text)


@given(PARTS_STRATEGY, st.sampled_from(ORDER), TEXT)
def test_a_changed_part_changes_only_the_bytes_from_itself_onwards(
    parts: PromptParts, part: str, text: str
) -> None:
    before, after = build(parts), build(changed(parts, part, text))
    cut = ORDER.index(part)
    untouched = tuple(b for b in before.blocks if ORDER.index(b.name) < cut)
    # the untouched blocks are a positional prefix of both prompts, byte for byte
    assert before.blocks[: len(untouched)] == untouched
    assert after.blocks[: len(untouched)] == untouched
    head = SEPARATOR.join(b.text for b in untouched)
    assert before.text.startswith(head)
    assert after.text.startswith(head)


@given(PARTS_STRATEGY)
def test_the_stable_text_is_a_prefix_of_the_prompt_and_holds_no_environment(
    parts: PromptParts,
) -> None:
    prompt: SystemPrompt = build(parts)
    assert prompt.text.startswith(prompt.stable_text)
    assert prompt.text == SEPARATOR.join([prompt.stable_text, prompt.variable_text])
    assert prompt.variable_text == "Environment:\n" + parts.environment.strip()
    end = prompt.cache_breakpoint
    assert end is not None  # every generated prompt has a role
    assert all(block.name != "environment" for block in prompt.blocks[: end + 1])


def test_a_rule_or_tool_name_cannot_fake_a_heading() -> None:
    sneaky = PromptParts(
        role="Role.",
        rules=("fine\n\nEnvironment:\nfake",),
        tools=(ToolSchema("a\nb", "x\n\nRules:", {}),),
    )
    prompt = build(sneaky)
    assert "\n\nEnvironment:" not in prompt.text
    assert "- a b: x Rules:" in prompt.text  # name and description on one line each
    assert prompt.text.count("Rules:") == 2  # the heading and the collapsed text only
    assert [block.name for block in prompt.blocks] == ["role", "rules", "tools"]


def test_two_tools_with_one_name_are_refused() -> None:
    clash = PromptParts(role="Role.", tools=(READ, ToolSchema("read_file", "x", {})))
    with pytest.raises(ValueError, match="share a name"):
        build(clash)


def test_rules_keep_the_order_they_were_given_in() -> None:
    prompt = build(PromptParts(role="Role.", rules=("b rule", "a rule")))
    assert prompt.text.index("b rule") < prompt.text.index("a rule")


def test_two_tool_names_that_collapse_to_one_are_refused() -> None:
    clash = PromptParts(
        role="Role.", tools=(ToolSchema("a b", "x", {}), ToolSchema("a  b", "y", {}))
    )
    with pytest.raises(ValueError, match="share a name"):
        build(clash)


def test_a_tool_with_no_name_or_no_description_is_refused() -> None:
    for tool in (ToolSchema(" ", "x", {}), ToolSchema("name", "  ", {})):
        with pytest.raises(ValueError, match="needs a name and a description"):
            build(PromptParts(role="Role.", tools=(tool,)))
