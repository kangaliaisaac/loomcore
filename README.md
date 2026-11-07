# loomcore

A small Python library that is an agent harness: the loop, tools, sandbox, context, guardrails and event log that sit around a model and turn it into an agent that can finish a task.

**Status: pre-release.** The design is written down ([intent.md](intent.md), [spec.md](spec.md)); the code is being built one component at a time. Nothing below is installable yet, and the usage notes describe what the library is for, not an API you can import today.

## What it is

An agent is a model plus a harness. The model reasons; the harness runs the loop, executes tools, keeps the model's context within budget, stops it when it should stop, and records everything. loomcore is that harness, kept small:

- a ReAct-style loop over the model's native tool use, written as a pure `step(state, event)` function;
- an append-only event log that is the only record of a run, so a run can be replayed, resumed or rewound;
- eight components, each usable without the driver: system-prompt builder, tool registry, sandbox boundary, filesystem workspace, context store, self-verification hook, guardrails with a human-approval seam, and observability with token-based cost.

## How to use it

When it ships, the shape of use is:

1. Write ordinary typed Python functions and register them as tools; their type hints become the schema the model sees.
2. Give a run its limits (turns, time, budget) and a task, and let the driver run it. Every step lands in the event log.
3. Read the log to see what happened, price it from the recorded token counts, or resume and rewind it.
4. Use a single component without the driver if you already have a loop: the registry, the sandbox, the guards and the context store can each be called from your own code.

A short hello-agent example, about twenty lines, will be added here with the driver.

## How not to use it

- **It is not a framework.** There is no agent DSL, no dependency-injection container and no global state. You pass things in.
- **The sandbox is a policy, not a virtual machine.** It denies by default and checks paths and commands, but it does not replace operating-system isolation. Do not run untrusted code on a machine you care about because loomcore is wrapped around it.
- **Allow-listing is on you.** A secret is out of reach of model-run code only until you allow-list the variable or path that holds it.
- **Tool output is untrusted data.** Never treat what a tool returns, such as a web page or a document, as instructions to the agent.
- **The event log is sensitive.** It holds prompts, responses and tool results. Redact before sharing it.
- **It does not plan, route or fan out for you in v0.1.** Planner-executor, script actions, model routing, sub-agents, best-of-N and tree search are named interfaces, not built.
- **No UI, hosted service or database.** Persistence is a JSONL file.

## References

The design rests on public work.

- Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629): interleave reasoning with actions and observations, and guard against loops. loomcore keeps the idea and replaces the parsed text format with native tool use.
- Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): start simple, make planning visible, put stopping conditions and human checkpoints in, and invest in tool design.
- Anthropic, [Scaling Managed Agents](https://www.anthropic.com/engineering/managed-agents): separate the model and loop, the tools and sandboxes, and the session log behind stable interfaces, and keep credentials out of the sandbox.
- Anthropic, [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents): one feature per session, a progress record, and end-to-end verification before something counts as done.
- Anthropic, [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps): treat each harness component as an assumption about what the model cannot do alone, and re-test it as models improve.
- Anthropic, [Building agents: when to fold subagents into your main agent loop](https://claude.com/resources/articles/building-agents-when-to-fold-subagents-into-your-main-agent-loop): prefer progressive disclosure over subagents unless an eval shows otherwise.
- HumanLayer, [12-Factor Agents](https://github.com/humanlayer/12-factor-agents): own your context window, treat tools as structured outputs, unify execution and business state, own your control flow.
- Fowler, [Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html), and Kleppmann, *Designing Data-Intensive Applications* (ch. 3 and 11): store the facts, derive state by replay, survive a torn write.
- Bernhardt, [Boundaries](https://www.destroyallsoftware.com/talks/boundaries): functional core, imperative shell.

## Licence

MIT. See [LICENSE](LICENSE).
