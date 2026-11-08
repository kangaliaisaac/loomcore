# loomcore

A small Python agent harness: the loop, tools, sandbox, context, guardrails and an append-only event log around a model. Read `intent.md` (what and why) and `spec.md` (the contract) first, then `plan.md` and `feature_list.json` for what to build next.

## Commands

```
uv sync                          # create .venv from the lock file
uv run --no-sync ruff check .    # lint
uv run --no-sync ruff format --check .
uv run --no-sync mypy src tests  # strict
uv run --no-sync pytest
./init.sh                        # sync, run all gates, print the next unfinished feature
```

## Rules

- Python 3.15, `src/` layout, full type hints, mypy strict. Interfaces are `Protocol`s; data is frozen dataclasses or pydantic models at the boundaries.
- Functional core, imperative shell. Logic is pure and takes its dependencies (model call, clock, filesystem, subprocess) as arguments; I/O sits at thin edges. No DI framework, no global state.
- No premature abstraction: write it concretely, abstract on the third repetition. Standard library first; a dependency needs a reason in the PR.
- Size limits (the linter enforces most): about 30 statements per function, at most 5 arguments, complexity at most 8, at most 4 returns, line length 88. One responsibility per module.
- Errors are values in the core, exceptions at the edge. Raise specific exceptions that say what to do; never a bare `except`.
- Comments say why, not what. Docstrings on public API only.
- Every behaviour has a test, written with the code. The scripted three-step eval runs in CI like a test.
- The spec is the contract. If the code needs to differ, change the spec in the same PR and say why.
- One feature per step, and each step ends mergeable. Take the next unfinished entry in `feature_list.json`; flip its `passes` to true only when its acceptance test is green.
- Never commit secrets, tokens or personal data. Commits use the GitHub noreply address.
- Branches are named `ikangalia/card-<number>/<topic>`. The owner reviews and merges every PR.
