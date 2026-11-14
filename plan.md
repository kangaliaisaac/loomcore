# loomcore: plan

[intent.md](intent.md) says what `loomcore` is for; [spec.md](spec.md) is the contract. This plan breaks the work into mergeable steps, one per card, in dependency order. Each step is one session's work, ends with the gates green, and takes its features from [feature_list.json](feature_list.json): flip a feature's `passes` to `true` only when its acceptance test is green. The plan names the steps; it does not restate the spec.

Before any step: `./init.sh` syncs the environment, runs the gates and prints the next unfinished feature.

## How each step ends

1. The feature's tests are written with the code and pass.
2. `ruff check`, `ruff format --check`, `mypy` (strict) and `pytest` are clean.
3. The spec is unchanged, or changed in the same PR with a reason.
4. One pull request from `ikangalia/card-<number>/<topic>`, reviewed once before the push; the owner reviews and merges.

## Steps

| Card | Step | Spec | Features | Mergeable when |
|---|---|---|---|---|
| 03 | Plan and repo setup (this step) | n/a | none | gates and CI run on an empty package; `plan.md` and `feature_list.json` exist |
| 04 | The event log | section 2, Rewind in section 1 | f01 to f06 | a log can be written, read back after a torn write, replayed to the same state, and a fork point can be judged clean (writing `run_forked` is card 05); the spend log works |
| 05 | The ReAct loop | section 1 | f07 to f11 | a scripted fake model drives a run to `finished` (`step` also covers crash and approval resumes and `AskHuman`); limits pause and resume; the ladder and `finish` work; resume repairs the spend log under the session lock, surfaces skipped spend lines and the spend-log path, and writes `run_forked` after `ensure_fork_point` (carried from card 04, f08) |
| 06 | Sandbox boundary and filesystem workspace | 3.3, 3.4 | f12, f13 | a secret in the parent is never passed to the container, which has no network and only the workspace writable; every path escape is refused |
| 07 | Tool registry | 3.2 | f14, f15 | typed functions become tools; conflicting calls run in order; timeouts and the cache behave (f15 in two parts: the cache, timeouts and planner first, then the driver running independent calls together) |
| 08 | Context store and system-prompt builder | 3.5, 3.1, plan tool | f16 to f18 | packing and compaction respect the budget; prompts are byte-stable |
| 09 | Guardrails and the human-in-the-loop seam | 3.7 | f19 | deny, redact and ask work and are logged; a run resumes from the log |
| 10 | Feedback, observability and cost | 3.6, 3.8, tokens and cost | f20, f21 | verify-then-reflect works; a recorded log is priced from tokens alone |
| 11 | Review: the scripted-task eval in CI | 1, 3, 5 | f22, f24 | the three-step eval runs in CI and the log is replayable; a hand-written loop completes a task with no driver |
| 12 | Ship and write-up | 6 | f23 | a second repository can import `loomcore` and run a multi-step task; README hello-agent example; the write-up |

## Order and why

The event log comes first because everything else reads or writes it, and the loop is a fold over it. The loop comes second so every later component can be exercised through a scripted run. The sandbox and workspace come before the tool registry because tools run inside them. The context store and prompt builder need the loop to feed. Guards sit between the loop and the tools, so they follow the registry. Observability and cost read the log, so they come last, and the eval ties everything together.

## Choices to keep

- `pricing.json` is a vendored, versioned table of public per-token prices; cost is always derived from recorded tokens.
- The anthropic SDK stays behind the `ModelCall` port. Core code never imports it; tests use a scripted fake.
- No feature is added that the spec does not name. The named-only interfaces (planner-executor, script actions, model routing, sub-agents, best-of-N, tree search, direct routing) stay `Protocol`s with a docstring and no implementation.
- Spec open items (the wording and test gaps listed in the spec PR) are fixed in the step that implements the rule, not in a separate spec pass.
