# `loomcore`

A small Python library that is an agent harness: the loop, tools, sandbox, context, guardrails and event log that sit around a model and turn it into an agent that can finish a task.

- [`loomcore`](#loomcore)
  - [What it is](#what-it-is)
  - [How to use it](#how-to-use-it)
  - [How not to use it](#how-not-to-use-it)
  - [Design](#design)
  - [References](#references)
  - [Licence](#licence)

**Status: pre-release.** The design is written down ([intent.md](intent.md), [spec.md](spec.md)); the code is being built one component at a time. Nothing below is installable yet, and the usage notes describe what the library is for, not an API you can import today.

## What it is

An agent is a model plus a harness. The model reasons; the harness runs the loop, executes tools, keeps the model's context within budget, stops it when it should stop, and records everything. `loomcore` is that harness, kept small:

- a ReAct-style loop over the model's native tool use, written as a pure `step(state, event)` function;
- an append-only event log that is the only record of a run, so a run can be replayed, resumed or rewound;
- limits that are part of the core: turns, time, output size and two budget caps (per run and per month). Reaching one pauses the run with the reason named, and you resume it with the limits you choose;
- eight components, each usable without the driver: system-prompt builder, tool registry, sandbox boundary, filesystem workspace, context store, self-verification hook, guardrails with a human-approval seam, and observability with token-based cost.

It is built first for four projects that import it (`filingstrawl`, `finanalyst`, `harnessbench` and `streamsense`), and for anyone who wants to wire a model to a few tools and get a run that ends on purpose, with a log they can read.

## How to use it

When it ships, the shape of use is:

1. Write ordinary typed Python functions and register them as tools; their type hints become the schema the model sees. Say which are `read_only`: only those are re-run without asking if a run is resumed after a crash.
2. Give a run its limits and a task, and let the driver run it. Turns, time and output size have defaults; the two budget caps have none, so you give each a value or say explicitly that it is unlimited. Every step lands in the event log.
3. Read the log to see what happened, and price it from the recorded token counts. A run that reaches a limit or needs an approval pauses; `resume` continues it. Rewinding forks a run from an earlier clean point; the fork starts its own turn, time and per-run spend counts, and only the monthly cap spans runs.
4. Use a single component without the driver if you already have a loop. The registry, the sandbox, the guards and the context store are designed to be called from your own code.

A short hello-agent example, about twenty lines, will be added here with the driver. [spec.md](spec.md) is the contract for every piece.

## How not to use it

- **It is not a framework.** There is no agent DSL, no dependency-injection container and no global state. You pass things in. It is a library you call, not a runtime that calls you.
- **The sandbox is a Docker container, not a virtual machine.** It denies by default and runs each command in a locked-down container (no network, read-only root, non-root user, resource limits), but a container shares the host kernel. It needs a running Docker daemon and an image you have pulled; it never pulls and never falls back to the host. Do not run untrusted code on a machine you care about because `loomcore` is wrapped around it.
- **Allow-listing is on you.** A secret is out of reach of model-run code only until you allow-list the variable or path that holds it.
- **Tool output is untrusted data.** Never treat what a tool returns, such as a web page or a document, as instructions to the agent.
- **The event log is sensitive.** It holds prompts, responses and tool results. A post guard can redact a secret from a tool result before it is logged, but only what the post guards you write are built to catch, and the log then holds the redacted content, not the secret; treat the log as sensitive and review it before sharing.
- **Rewind restores conversation state, not files.** What the old run wrote to the workspace after the fork point is still on disk, and the fork starts from that workspace.
- **A tool timeout abandons the call; it does not kill it.** Python cannot stop a running thread, so a timed-out tool may keep running and write late. A tool that writes should run through the sandbox, which kills its container on timeout.
- **One writer per session file.** Do not point two processes at the same session; the second waits or fails.
- **Among the optional extras, version 0.1 ships a plan tool (the model's latest plan is written to a file and kept in context, not a planner) and verify-then-reflect.** Planner-executor, script actions, model routing, sub-agents, best-of-N, tree search and direct routing are named interfaces with no implementation, so do not expect them to run.
- **No UI, hosted service or database.** Persistence is a JSONL file.

## Design

[intent.md](intent.md) says what `loomcore` is for and what it will not do. [spec.md](spec.md) is the contract: the loop, the event log, the eight components and the tests that prove them. The sources for each named interface are linked from the spec.

## References

The design rests on public work.

- Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629): interleave reasoning with actions and observations, and guard against loops. `loomcore` keeps the idea and replaces the parsed text format with native tool use.
- Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): start simple, make planning visible, put stopping conditions and human checkpoints in, and invest in tool design.
- Anthropic, [Scaling Managed Agents](https://www.anthropic.com/engineering/managed-agents): separate the model and loop, the tools and sandboxes, and the session log behind stable interfaces, and keep credentials out of the sandbox.
- Anthropic, [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents): one feature per session, a progress record, and end-to-end verification before something counts as done.
- Anthropic, [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps): treat each harness component as an assumption about what the model cannot do alone, and re-test it as models improve.
- Anthropic, [Building agents: when to fold subagents into your main agent loop](https://claude.com/resources/articles/building-agents-when-to-fold-subagents-into-your-main-agent-loop): prefer progressive disclosure over subagents unless an eval shows otherwise.
- Barbaste et al., [Harness Engineering: Anatomy, Architecture, and Evolution of Coding Agents](https://arxiv.org/abs/2609.00006): a source-code study of eleven coding-agent systems, cited for the anatomy and architecture of a harness.
- HumanLayer, [12-Factor Agents](https://github.com/humanlayer/12-factor-agents): own your context window, treat tools as structured outputs, unify execution and business state, own your control flow.
- Fowler, [Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html), and Kleppmann, *Designing Data-Intensive Applications* (first edition, ch. 3 and 11): store the facts and derive state by replay. Repairing a torn final line after a crash is `loomcore`'s own rule.
- Bernhardt, [Boundaries](https://www.destroyallsoftware.com/talks/boundaries): functional core, imperative shell.

## Licence

MIT. See [LICENSE](LICENSE).
