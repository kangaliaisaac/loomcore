# loomcore: intent

## What it is

loomcore is a small Python library that is a harness. A model reasons; the harness does everything else a working agent needs: the loop, the tools, the sandbox, the context, the guardrails and the record of what happened. loomcore owns that plumbing so the code that uses it can stay about the task.

It runs a ReAct-style loop over the model's native tool use (`tool_use` and `tool_result` blocks, `stop_reason`, extended thinking). The loop is a pure function, `step(state, event) → (state, effects)`, and the trajectory is an append-only event log kept outside the model's context. State is a fold over that log, so any run can be replayed, resumed or rewound. The same pieces run a fixed workflow as readily as an open-ended loop. [spec.md](spec.md) is the contract.

## Who it is for

- **The four projects built on it:** a retrieval engine, an analysis agent, an evaluation and observability suite, and a streaming detector. Each imports loomcore instead of re-implementing the plumbing.
- **An outside user** who wants to wire a model to a few tools and get a run that ends on purpose, with a log they can read.

## What it commits to

- **Eight components, each usable on its own:** system-prompt builder, tool registry, sandbox boundary, filesystem workspace, memory and context store, feedback and self-verification hook, guardrails with a human-in-the-loop seam, and observability. Every one works without loomcore's driver, so a user can bring their own loop.
- **Nothing hidden.** Raw prompts, model responses and tool results are in the log, so a run can be debugged by reading it.
- **Credentials stay out of the sandbox.** Tools run with an explicit allow-list of paths, commands and environment variables, deny by default; no secret is reachable from code the model can run unless the user explicitly allow-lists it.
- **One source of truth.** The event log is the only record. Everything that reports on a run reads it; none reads another reader's output. Token counts are stored, and cost is derived from them.
- **Runs end on purpose.** A typed `finish(result)` tool, a graceful stop at the turn, budget or time limit that says why and where to resume, and an escalation ladder when the model loops: nudge, switch strategy, ask a human, stop with the best partial result.
- **Extensions plug into seams that already exist** (a tool, a hook around the step function, a stop policy, or an alternative driver). The core does not change, and each extension is off by default.
- **Functional core, imperative shell.** Logic is pure; model calls, disk, clock and subprocesses sit at thin edges and are passed in.

## What it will not do in v0.1

- Plan, route or fan out for you. Planner-executor, script actions, model routing, sub-agents, best-of-N and tree search are named interfaces in the spec, not built.
- Be a framework: no agent DSL, no dependency-injection container, no global state.
- Ship a UI, a hosted service or a database. Persistence is the JSONL event log.

## Done means

A second repository can `import loomcore`, register a tool and run a multi-step task to completion, with the full event log persisted and replayable. Along the way:

- every component has a contract and a test;
- a scripted three-step task runs as an eval in CI;
- token and cost accounting is native to the library;
- the README carries a hello-agent example of about twenty lines;
- a short write-up says why the harness, not the model, is the engineering.
