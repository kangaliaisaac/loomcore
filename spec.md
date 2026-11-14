# `loomcore`: spec

- [`loomcore`: spec](#loomcore-spec)
  - [1. The loop](#1-the-loop)
    - [Ending a run](#ending-a-run)
    - [Limits](#limits)
    - [The finish tool](#the-finish-tool)
    - [Escalation ladder](#escalation-ladder)
    - [Extension seams](#extension-seams)
  - [2. The event log](#2-the-event-log)
    - [Envelope](#envelope)
    - [Kinds](#kinds)
    - [Writing, reading, replay](#writing-reading-replay)
    - [Tokens and cost](#tokens-and-cost)
    - [Run spend line](#run-spend-line)
  - [3. The eight components](#3-the-eight-components)
    - [3.1 System-prompt builder (`prompt.py`)](#31-system-prompt-builder-promptpy)
    - [3.2 Tool registry (`tools.py`)](#32-tool-registry-toolspy)
    - [3.3 Sandbox boundary (`sandbox.py`)](#33-sandbox-boundary-sandboxpy)
    - [3.4 Filesystem workspace (`workspace.py`)](#34-filesystem-workspace-workspacepy)
    - [3.5 Memory and context store (`context.py`)](#35-memory-and-context-store-contextpy)
    - [3.6 Feedback and self-verification hook (`verify.py`)](#36-feedback-and-self-verification-hook-verifypy)
    - [3.7 Guardrails and the human-in-the-loop seam (`guards.py`)](#37-guardrails-and-the-human-in-the-loop-seam-guardspy)
    - [3.8 Observability (`observability/`)](#38-observability-observability)
  - [4. Layout](#4-layout)
  - [5. Tests and the eval](#5-tests-and-the-eval)
  - [6. Done for v0.1](#6-done-for-v01)


The contract for v0.1. [intent.md](intent.md) says what `loomcore` is for; this file says what it does, component by component. Where the two differ, this file wins. 

Python 3.15, `src/` layout, full type hints, mypy strict. Interfaces are `Protocol`s; data is frozen dataclasses; effects sit at the edge and are passed in.

## 1. The loop

`loomcore` runs a [ReAct](https://arxiv.org/abs/2210.03629)-style loop over native tool use: 
- the model emits `tool_use` blocks, 
- the harness runs them and returns `tool_result` blocks, 
- and `stop_reason` says when the turn is over. 

The paper's parsed "Thought / Action / Observation" text is not used; extended thinking and tool blocks replace it.

The core is one pure function:

```python
def step(
    state: RunState, event: Event, read_only: Callable[[str], bool] = ...
) -> tuple[RunState, list[Effect]]: ...   # read_only(tool name); an unknown tool is not read-only
```

1. `step` never performs I/O. A thin driver feeds it events, performs the effects it returns (call the model, run a tool, ask a human, stop) and appends each result to the log as the next event. 
2. `RunState` is a fold over the log: `functools.reduce(apply, events, initial)`, with `apply` pure (it counts the run's own `turn_started` events in `turns`, and a fork starts again at 0, and so does the `turn` its first `turn_started` carries: the fork's turns number from 1). Given the same configuration, a new process can load the log and continue: the configuration (model, system prompt and tool schemas) is logged in `run_started` with its digest, and again, with its digest, on each switch-strategy `escalation`; `resume` checks two things and refuses to continue if either differs: the configuration the resuming process holds must match `run_started.config_digest` (for a fork, the root's), and the configuration it rebuilds from the latest switch must match that switch's `config_digest`. The latest configuration is this run's own latest switch, else the one in force on its parent chain at `from.seq`, else its root's `run_started`. Limits are logged beside the configuration but are not part of the digest, so they can be changed on resume.

```python
# Effect = CallModel | RequestCalls | RunTool | Fail | AskHuman | Stop
#   RequestCalls: write the turn's tool_requested lines together, then judge them
#   RunTool may carry a ladder note; Fail completes a call ok=false, unrun

class Driver(Protocol):                 # bound to one session file and one spend log
    def run(self, task: Task) -> RunResult: ...            # Task = (prompt, limits)
    def resume(
        self, 
        ref: RunRef, 
        decision: Decision | None = None,
        limits: Limits | None = None) -> RunResult: ...   # RunRef = (session_id, run_id); limits required after a limit pause
    def fork(
        self, 
        origin: ForkOrigin, 
        hint: str | None = None,
        limits: Limits | None = None) -> RunResult: ...   # checks the point, then writes run_forked
```

Every component below works without the driver: a user may call the registry, the sandbox, the guards and the context store from their own loop.

**Ports** (the only places that touch the world, injected into the driver):

```python
ModelCall = Callable[[ModelRequest], ModelResponse]       # wraps the Anthropic SDK; tests use a scripted fake
Clock = Callable[[], datetime]                            # wall-clock UTC
Monotonic = Callable[[], float]                           # durations
Random = Callable[[], float]                              # jitter; tests pass a fixed one
```

The driver also takes the tool port `(name, input) → ToolResult` (it never raises and has its own timeout), `read_only(name) → bool`, `sleep(seconds)`, `new_run_id()`, and a `Pricing` (`worst_case(request)`, `cost(message) → (usd, pricing_version)`; the vendored table arrives with the cost component). The model port raises `ModelCallError(CallError)` when a call fails; the retry policy reads it. The driver is built from a `Config` (model, system prompt, tool schemas), whose digest is the one in `run_started`, and refuses a spend log whose directory does not exist, because a mistyped path would read as empty. It logs the spend-log path it uses and every spend line it cannot read.

### Ending a run

A run ends on purpose. `run_finished.result` is the model's final text for `end_turn` and `refused`, and the message for `error`. `run_finished` is terminal and carries the reason. A run that waits for a human, or that reached a limit, writes `run_paused` instead and is later resumed, so a run has at most one `run_finished`. A paused run that is never resumed is simply stopped.

| Event | Reason | Meaning |
|---|---|---|
| `run_finished` | `finished` | the model called the typed `finish(result)` tool and verification passed |
| `run_finished` | `end_turn` | the model stopped without `finish` and no tool was pending; the verifier runs here too, and a failure continues the loop |
| `run_finished` | `refused` | the model's `stop_reason` was `refusal` |
| `run_finished` | `stuck` | the escalation ladder ran out |
| `run_finished` | `killed` | an external signal or the kill switch; a budget cap pauses with `limit`, not `killed` |
| `run_finished` | `error` | an unrecoverable error, with the message |
| `run_paused` | `awaiting_approval` | a guard, the escalation ladder, or crash resume of a tool that is not read-only asked for approval; `resume` with a decision appends `run_resumed`, then `approval_granted` or `approval_denied`, and continues the same run: a grant runs every pending call of the turn; the driver appends a grant only when none of those calls has run (the ladder asks before any runs) or after a crash resume has settled them, so a call never runs twice; a denial completes the asked-about call with `ok=false` and `denied by <by>: <note>`, writes that `Fail` before anything runs, and runs the rest as a grant would (with the rung's notes); if a crash then interrupts a later call of a tool that is not read-only, the re-run question is asked about it, as for a grant. `resume_from` on `run_paused` and `run_resumed` is the seq of the run's last event before the gap. `resume` without a decision on such a pause is an error, and so is a decision on a run that is not paused for approval |
| `run_paused` | `limit` | `max_turns`, `timeout`, `run_cap`, `month_cap` or `max_output_tokens` was reached; the stop is graceful and names which one in `limit`. `resume(ref, limits=...)` must be given and appends `run_resumed` with those limits; the turn count, the spend and the timeout segments carry over. The limits need not be raised: the next `check_limits` decides (for `max_output_tokens`, the `run_resumed` has reset `truncations`, so the run gets fresh attempts), and if a limit is still reached the run pauses again with the same `limit` (a run paused on `month_cap` on the 31st has room again on the 1st). `resume` on a finished run is an error |

Before `run_finished` is written for any reason, including `killed` and `error`, the driver completes every still-pending `tool_use` with `ok=false` and the reason as `content`. A process that dies before it can do so is an ordinary crash: `resume` settles the pending calls as it does after any crash (a `read_only` tool is re-run, any other tool becomes an `Ask`), not by completing them `ok=false`. If the turn's `escalation` event is already logged, `resume` finishes that rung instead: after `stuck` it completes the pending calls `ok=false` and writes `run_finished`; an `approval_requested` with no decision and no `run_paused` is treated as `awaiting_approval`, and `resume` writes the `run_paused` (after its `run_resumed`). After a crash, `resume` appends `run_resumed` before any other event (the `run_paused` it adds for an open approval, the re-run of a read-only tool), because the open segment of the timeout ends at the last event before `run_resumed`: an event written first at the restart would count the downtime as active time. A kill is recorded only by its `run_finished`, so a kill interrupted by a crash is not in the log and a resumed run continues; kill it again.

Other model stop reasons are handled and logged: 
- `tool_use` continues the loop, 
- `stop_sequence` is treated as `end_turn`, 
- `max_tokens` (the model hit the request's output ceiling, `max_output_tokens`) continues once and then, if the next call is cut off too, pauses with the limit `max_output_tokens` (counted in `RunState.truncations`, below). To continue, the partial assistant turn is sent back with a short instruction to carry on; a `tool_use` block cut off mid-input (the last block of a response whose `stop_reason` is `max_tokens`, when that block is a `tool_use`) is discarded, never run, and the model is asked again, while complete `tool_use` blocks before it are run as normal (each with its `tool_requested` and one `tool_completed`) and their results go back with the instruction; the cut-off block is excluded from the rule that every `tool_use` gets a `tool_completed` and from the clean-fork-point rule, and 
- `pause_turn` resumes by sending the response back, and
- `tool_use` with no complete `tool_use` block to run, and any other or missing `stop_reason`, end the run with `error`, naming the value; a response with complete `tool_use` blocks runs them whatever its `stop_reason`.

### Limits

Turn, time and budget limits are part of the core, not extensions. `max_turns` and `timeout` have defaults (50 turns, 10 minutes), and so do `max_output_tokens`, the per-request output ceiling (8192 tokens), `repeat_window` (10 calls) and `repeat_threshold` (3 identical calls), which the escalation ladder reads. The budget has two caps, `run_cap` (this run) and `month_cap` (all runs this UTC month); neither has a default, and each must be given or explicitly set to `None` for unlimited. The check pauses the run, naming `run_cap` or `month_cap` as the limit, when that cap would be crossed. Before each model call the driver checks the limits as a pure function over the state: `check_limits(state, limits, month_to_date, worst_case, now) → WithinLimits | LimitReached(limit)`. The `limits` passed in are the ones in force for this run: its latest `run_resumed` that carries `limits`, else its `run_forked.limits`, else the limits in force on its parent chain at `from.seq`, else its `run_started.limits`. One session file holds many runs, so a sibling's override never applies to this run. `max_output_tokens` is enforced by `check_limits` too. `RunState` holds `truncations`, the number of consecutive `model_call` events that stopped on `max_tokens`: any other `model_call` resets it to 0, and so do `run_forked` (a fork starts fresh, whatever its parent's streak was) and a `run_resumed` that follows a pause whose limit was `max_output_tokens`. A crash or approval resume keeps it, so a run that died after its second truncated call still pauses. The check returns `LimitReached("max_output_tokens")` when `truncations` is 2 or more, so the rule is a pure fold over the log and a resumed process rebuilds it.

- **The comparisons.** `check_limits` returns the first of these that holds, in this order: `max_output_tokens` (`truncations` is 2 or more), `max_turns` (`turns` has reached it, so 50 turns run and the 51st call pauses, because the driver checks before it writes the next `turn_started`; every model call, a continuation after `max_tokens` or `pause_turn` included, is its own turn), `timeout` (active time has reached it), `run_cap` and `month_cap` (the total would be more than the cap: spending exactly the cap is allowed). `Limits` holds `timeout` in seconds, and a naive `now` is refused, because months are judged in UTC. `RunState` keeps the run's own spend by UTC month, so both caps read from the fold; a fork starts it empty.
- **Per-run spend** is a fold over the `model_call` events of this run's chain; `run_events` selects the chain, and `run_forked` starts the total at zero, so a sibling's calls and a parent's are never charged to it. A forked run inherits its parent's events for context but not for spend.
- **`month_to_date`** is read from the spend log at the edge and passed in: the sum of entries whose `at` falls in the current UTC month, excluding this run's own lines. The check compares per-run spend plus `worst_case` with `run_cap`, and `month_to_date` plus per-run spend plus `worst_case` with `month_cap`, so nothing is counted twice across a pause. For the `month_cap` comparison, this run's spend counts only its `model_call` events in the current UTC month, so a run that spans a month boundary does not charge last month's spend to this month.
- **`worst_case`** prices the next call as its known input tokens plus `max_output_tokens` of output (the per-request ceiling that every `ModelRequest` carries, taken from `Limits`), at the request model's prices.
- **`timeout`** counts active time, as a sum of segments. A segment starts at `run_started`, `run_forked` or `run_resumed` and ends at the last event before the next gap (a `run_paused`, or the end of the log after a crash); the open segment ends at `now` (the `Clock` port). A run waiting an hour for approval, or resumed hours after a crash, does not time out for the time it was down.

**Retries** are a pure policy, `(attempt, error, random) → Delay | GiveUp`, with the random source passed in; the error is a `CallError` (kind `http`, `connection` or `timeout`, the status, `retry_after` seconds) so the policy knows no SDK type, and the defaults are a 0.5 s base, a 30 s cap and 5 retries after the first call. 429 and 529, other 5xx, connection errors and timeouts get exponential backoff with full jitter (`random() * min(cap, base * 2**attempt)`) and honour `retry-after` as a floor on the delay (it is not capped, so the driver checks the run's `timeout` before it sleeps: if the sleep would reach it, the run pauses with `timeout` instead); any other 4xx is never retried. Every model and tool call has a timeout. A tool timeout abandons the call: it is logged as failed with `timeout` and the loop moves on, but Python cannot kill a thread, so the call may keep running. A tool that writes must therefore run through the sandbox (a Docker container, killed on timeout) or accept a cancel event; a late write from an abandoned call still bumps the workspace generation counter, so the cache stays correct.

**Two identifiers, two jobs.** A replay or a retry does not re-run a completed call: tool completions are matched by `tool_use_id`, and if the log already holds a `tool_completed` for it, the result is reused. After a crash, at most one call is interrupted: calls run one at a time, in order, and only once every `tool_requested` line of the turn is written, so the only call that can have started is the first one with a `tool_requested` and no `tool_completed`, and none can have while a line is missing or while the ladder holds the turn for a human whose answer is not logged. The rule is conservative on purpose: a crash just after an `approval_granted`, before the first call runs, still counts the first pending call as interrupted, so a call that never started may get the re-run question; a wasted question is safer than a second write. Delivery is at-least-once. If the process dies after a tool runs but before its completion is appended, `resume` re-runs a tool registered `read_only=True`, and turns a call to any other tool, declared or not, into an `Ask` before running it again (one `approval_requested`, about the interrupted call). A tool that does not say it is read-only is treated as one that has side effects. Separately, `call_hash` is `sha256(json.dumps({"tool": name, "input": args}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()`, stored on `tool_requested`; it identifies a repeated call for the repeat detector and the cache. A deliberate repeat (running the tests again after an edit) is never refused at the nudge or switch rungs, but enough identical calls within `repeat_window` still reach the ask and stop rungs, because the detector cannot see that an edit came between them.

### The finish tool

`finish(result)` is a typed tool whose schema is `finish.SCHEMA`; the configuration lists it among the tools. When a turn calls it, the run ends with reason `finished` and the call's `result` as `run_finished.result`, even when that is null. The `finish` call is completed with `ok=true`; any other call in the same turn is completed with `ok=false` and is not run, so "every `tool_use` gets exactly one `tool_completed`" holds. The verifier (section 3.6) runs before the run ends once it exists; until then the claim stands.

### Escalation ladder

The repeat detector looks at the run's last `repeat_window` tool calls (default 10), the one being judged included, and counts those with the same `call_hash`. When that count reaches or exceeds `repeat_threshold` (default 3) the call trips the detector, so each further identical call trips it again and the ladder can climb. The detector is exact: it sees identical calls only, so a model that changes a harmless argument each time is not caught, and `max_turns`, `timeout` and `run_cap` are the backstop for near-repeats. It is a fold over the `tool_requested` lines of the run's own calls and the `approval_granted` lines that answer its own ladder asks (the same `tool_use_id` as the `escalation`) (a fork counts only its own), judged once per turn in emission order, so it needs no workspace state and replays the same way every time.

If one or more calls trip in a turn, the ladder climbs one rung for that turn, however many tripped: nudge (a short corrective message), switch strategy (a different prompt or tool set; with no alternative configuration given, the rung only adds its message), ask a human (an `Ask`), stop with the best partial result (`stuck`). The rung belongs to the run, is part of `RunState`, never goes down, and starts at 0 in a fork. Each rung is an `escalation` event.

The calls that tripped are pending `tool_use` blocks, and the rung settles them, so "every `tool_use` gets exactly one `tool_completed`" holds on every path:

- **Nudge and switch strategy:** the calls run as normal, and the rung's message is added to the `content` of each call that tripped (the corrective nudge, or the instruction that the approach is changing), after the post guards have run and before `tool_completed` is written, so the log holds exactly what the model sees. Nothing is blocked, so a deliberate repeat is never refused. Switch strategy changes the prompt or tool set for the turns that follow, logs the new configuration and its digest on its `escalation` event, so a new process can rebuild it from the log.
- **Ask a human:** the question is about the first call that tripped, and `approval_requested`, `approval_granted` and `approval_denied` carry its `tool_use_id`. None of the turn's calls runs before the decision. On a grant they all run as normal, and calls made before the grant no longer count toward that `call_hash`; the rung does not drop, so the next trip goes to `stuck`, because a human approved the call once, not for good. On a denial the call that was asked about is completed with `ok=false` and the denial note, and the turn's other calls run as normal.
- **Stop (`stuck`):** every still-pending `tool_use` in the turn is completed with `ok=false` and the reason before `run_finished` is written.

After a crash, `resume` finishes the rung: `stuck` stops; an `ask` with no decision asks again and runs nothing; a decision already logged is settled and its question is not asked again (a denied call fails; a granted turn runs, except that a call a crash left possibly run, of a tool that is not read-only, is asked about first with the re-run question, because the human answered the ladder's question and not that one); a nudge or a switch runs the calls with their notes, except that the call a crash left possibly run is asked about first (the grant then runs the turn with its notes). The `escalation` is written before the turn's `tool_requested` lines, so a turn whose lines are all written and that has no `escalation` was judged clean; a turn with only some lines written and no `escalation` is judged again as a whole, and a turn already judged is not judged twice. A decision stays in force for its whole turn, so a crash after the asked call completed does not ask again, and a denial still fails only the call it was asked about, whatever the crash interrupted later. After a crash that follows a `finish` call or a `stuck` rung, `resume` writes the `run_finished` without running anything.

A fork after any of these is a clean point once the completions are written.

### Extension seams

An extension plugs into a seam that already exists and never changes the core. Each is off by default and enabled by configuration.

| Seam | Examples |
|---|---|
| A tool | the plan tool (a plan file always kept in context), `finish`, note-taking |
| A route matched before `CallModel` | direct dispatch |
| A hook around `step` | verify then reflect, a periodic critique, replan on failure |
| A stop policy | the escalation ladder, custom stop conditions |
| An alternative driver reusing the core | bring your own loop |

**Built in v0.1:** the escalation ladder, `finish` and graceful stop, the plan tool, verify then reflect, rewind, bring-your-own-loop. 

**Named interfaces only, not built.** Each is a `Protocol` with a docstring and no implementation. The links are the public work each one is named after:

- `PlannerExecutor`: plan once, run the tools with no model call in between, then one call to combine the results. 
  - Xu et al., [ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models](https://arxiv.org/abs/2305.18323).
- `ScriptAction`: the model acts by writing code, not one tool call at a time. 
  - Wang et al., [Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2402.01030) (CodeAct).
- `ModelRouter`: pick a model per request. 
  - Ong et al., [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665).
- `SubAgent`: hand a piece of the task to a separate agent with its own context. 
  - Anthropic, [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).
- `BestOfN`: sample several attempts and keep the best. 
  - Brown et al., [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787).
- `TreeSearch`: explore branches of actions, score them and back up the values. 
  - Zhou et al., [Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/abs/2310.04406) (LATS); 
  - Yao et al., [Tree of Thoughts](https://arxiv.org/abs/2305.10601) for the search over reasoning steps it builds on.
- `DirectRoute`: a request that matches a known route runs plain code, with no model call. 
  - Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (the routing workflow).

**Rewind** forks a run: a new `run_id` in the same session file (a fork names a run, not a session, and run ids are unique within the file) starts with a `run_forked` event holding `from: {runId, seq}` and optionally a human hint, then continues; a hint is sent to the model as the next user message. No events are copied, so replaying the new run reads the old run's events up to and including `from.seq` and then its own, and every model call is counted once. `from.seq` must be a clean point, judged on the state folded up to and including it: it comes before the parent's `run_finished` if it has one; every `tool_use` block in every `model_call` up to it (other than a block cut off by `max_tokens`) has a `tool_completed` (so a point after a `model_call` whose `tool_requested` lines are not yet written, or after only some of a turn's calls, is not clean); every `approval_requested` has an `approval_granted` or `approval_denied`; and every `run_paused` has a later `run_resumed`. The driver refuses any other point, so a fork never starts paused, awaiting approval, or with a tool call pending. The old run is untouched. A fork has no `run_started` of its own: its configuration is the one in force on its parent chain at `from.seq` (the latest switch-strategy configuration on the chain, else the root's `run_started`), and `resume` compares that digest. Replay follows `from` recursively. The repeat detector counts only the fork's own calls, like spend, so a fork gets a fresh start from that point. Its limits are the latest in force on the chain at `from.seq`, unless `run_forked` carries a `limits` override. `max_turns`, `timeout` and `run_cap` apply per run, and a fork is a new run, so rewind deliberately starts those counts again; only a person or calling code forks, never the loop itself, and `month_cap` binds across every run. Its `timeout` clock starts at its `run_forked` timestamp and its turn count starts at 0, so neither includes the parent's time or turns, and its per-run spend counts only its own `model_call` events. Rewind restores conversation state, not files: what the old run wrote after `seq` is still on disk, and the fork starts from that workspace.

## 2. The event log

The log is append-only JSONL, one event per line, and the only record of a run. Each session is one file, `<runs-root>/<project>/<sessionId>.jsonl`, holding every run of that session, so `(session_id, run_id)` locates a run. Everything that reports on a run reads it; nothing reads another reader's output.

### Envelope

Every line carries:

| Field | Type | Meaning |
|---|---|---|
| `schema` | str | `"loomcore.event.v1"`; readers accept older versions |
| `seq` | int | 0, 1, 2…, gap-free within a run |
| `timestamp` | str | wall-clock UTC, ISO 8601 with milliseconds |
| `sessionId` | str | the session the run belongs to |
| `runId` | str | the run; a rewind starts a new one |
| `kind` | str | the event kind (below) |

The field names `sessionId`, `timestamp` and the `message` block below follow the transcript format that existing Claude Code tooling already parses, so a `loomcore` log needs a new root directory and no new parser. Files live at `<runs-root>/<project>/<sessionId>.jsonl`, with sub-run logs under `subagents/`.

### Kinds

| Kind | Emitted by | Payload (besides the envelope) |
|---|---|---|
| `run_started` | driver | `task`, `config` (model, system prompt, tool schemas), `limits`, `config_digest` (over `config` only) |
| `run_forked` | driver | `from` (`runId`, `seq`), `hint`, `limits` (optional override) |
| `turn_started` | driver | `turn` |
| `model_call` | driver | `type: "assistant"`, `message`, `duration_ms`, `cost_usd`, `pricing_version` (below) |
| `tool_requested` | driver | `tool_use_id`, `name`, `input`, `call_hash`; the turn's calls are judged by the repeat detector first, then the turn's lines are written together (the `escalation`, if any, before them), and only then does any call run; a `finish` call has none |
| `tool_completed` | driver | `tool_use_id`, `ok`, `content`, `duration_ms`, `cached` (a completion for a call that did not run has `duration_ms` 0 and `cached` false) |
| `guard_decision` | guards | `stage` (`pre`/`post`), `tool_use_id`, `decision` (`allow`/`deny`/`ask`/`redact`; `ask` only at `pre`, `redact` only at `post`), `reason` |
| `approval_requested` | guards, driver | `tool_use_id`, `question` |
| `approval_granted`, `approval_denied` | driver | `tool_use_id`, `by`, `note` |
| `sandbox_denied` | sandbox, workspace | `tool_use_id` (when a tool call caused it), `action`, `target`, `reason` |
| `context_packed` | context store | `selected`, `dropped`, `tokens`, `budget` |
| `context_compacted` | context store | `strategy`, `tokens_before`, `tokens_after` |
| `verification_passed`, `verification_failed` | verifier | `verifier`, `message` |
| `escalation` | driver | `rung`, `reason`, `tool_use_id` and `call_hash` (of the first call that tripped), `tripped` (every call that tripped), `config` and `config_digest` (on switch strategy) |
| `note_written` | workspace | `path` |
| `run_paused` | driver | `reason` (`awaiting_approval` or `limit`), `limit` (which one, when the reason is `limit`), `resume_from` |
| `run_resumed` | driver | `reason` (`approval`, `limit` or `crash`), `resume_from`, `limits` (the limits in force from here on for this run; required after a `limit` pause) |
| `run_finished` | driver | `reason`, `result` |

The `model_call` line has the same `usage` block the Anthropic Messages API returns:

```json
{
    "schema": "loomcore.event.v1", 
    "seq": 7, 
    "kind": "model_call", 
    "type": "assistant",
    "timestamp": "2026-11-03T08:14:02.311Z", 
    "sessionId": "run-2026-11-03-0814", 
    "runId": "r1",
    "message": {
        "id": "msg_01…", 
        "model": "claude-opus-5-5",
        "usage": {
            "input_tokens": 1840, 
            "output_tokens": 312,
            "cache_read_input_tokens": 9200, 
            "cache_creation_input_tokens": 0,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 0, 
                "ephemeral_1h_input_tokens": 0
            }
        },
        "content": [
            {
                "type": "tool_use", 
                "name": "retrieve_passages"
            }
        ]
    },
    "cost_usd": 0.0129, 
    "pricing_version": "2026-10"
}
```

### Writing, reading, replay

- The writer appends one line and flushes; an `fsync` option exists for durability-sensitive runs. A session file has one writer at a time: the writer holds an exclusive advisory lock (`fcntl.flock` on `<sessionId>.lock`), which the operating system releases when the holder dies, so a crash never leaves a stale lock; a second process that wants to write waits or fails, so a fork cannot run while its parent is resumed. Before its first append to an existing file the writer, holding the lock, cuts the file back to the last complete newline, so a torn final line from a crash never ends up in the middle of the log, and keeps the bytes it cut as `cut_tail` on the writer so a resume can report them. If an append fails part-way (a full disk, an interrupt) or its `fsync` fails, the writer cuts the line back to where the append began before it raises, so a failed append leaves no line and a retry cannot repeat a `seq`; if the cut-back itself fails the writer refuses further appends.
- There is no separate log redactor. A secret in a tool result is removed by a post guard (`Redact`, section 3.7) before `tool_completed` is written, so the log holds what the model saw (the typed `finish` call is the one exception, section 3.2).
- The reader yields events in order. An unterminated final line (a torn write: each event is one write ending in a newline, so a crash can only leave a line without it) is dropped and reported, whether or not it parses; any terminated line that does not parse is an error, the last one included.
- Snapshots of derived state are written to a temporary file and moved with `os.replace`, so they are atomic, and the file and then its directory are fsynced so the rename survives a power loss. Replay may start from a snapshot, which records how many events it folds in and a sha256 of the last one's line (so a log that lost its tail and regrew differently is caught, not only a different run); it also records a version number (bumped when `RunState`'s meaning changes) and the names of `RunState`'s fields, so a snapshot written before a field was added or a meaning changed is refused, and a file that does not validate raises `StaleSnapshotError`. A snapshot that does not match the chain it is applied to is refused as stale, not replayed.
- Replay is deterministic: the same log always yields the same state.

### Tokens and cost

Token counts are the source of truth and cost is derived. `cost_usd` on an event is advisory; reports re-price from tokens with the versioned table named by `pricing_version`. The table is a vendored `pricing.json`. Money is `decimal.Decimal`.

### Run spend line

After each `model_call`, `loomcore` appends one line to a spend log: `{"at", "usd", "run", "seq", "note"}`, where `run` is `"<sessionId>/<runId>"` and `seq` is the `model_call`'s seq in that run and `usd` is the cost of that one call. Writing per call, not at the end, means a crashed run and a run still in progress are both already in the log, so the month cap sees them. Each record is written with one `os.write` on an `O_APPEND` descriptor, as `"\n" + json + "\n"`, so concurrent runs can share the file without the session lock. The leading newline means a torn append from a crash can never absorb the next run's line: the next record starts on a fresh line. Each line stands alone, so the reader ignores blank lines and skips and reports a line that fails to parse; a missing file reads as empty, and the first append creates the log's directory (the first run has no state yet); so a mistyped path reads as no spend and the driver logs the path it reads. The line format gained `seq` before any release, so lines without it are not read (they are skipped and reported); there is no migration. A skipped line is spend the cap cannot see, so the driver reads the month's spend through `month_spend`, which returns the total together with `skipped`, and surfaces them (a warning in the run's output) instead of dropping them silently. `month_to_date` for `check_limits` is read from this log as defined in section 1: the current UTC month, excluding the run's own lines. If a process dies between a `model_call` and its spend line, `resume` writes, first, a line for each of the run's `model_call` events that has none (found by `run` and `seq`), dated at the call so it stays in the month it was made. Concurrent runs can overshoot a cap by at most one call each, because the worst case is priced before every call but another run may spend in the meantime. A crash that tears the `model_call` line itself loses that call from both the log and the spend log, so the month total can miss at most one paid call per crash.

## 3. The eight components

Each component is a module with its own `Protocol`, a one-line contract, the events it emits and the tests that prove it.

### 3.1 System-prompt builder (`prompt.py`)

```python
class PromptBuilder(Protocol):
    def build(self, parts: PromptParts) -> SystemPrompt: ...
```

Deterministic assembly from parts (role, rules, tool docs, environment) in a fixed order, with nothing time-varying in the stable prefix. The result marks the cache breakpoint on its last stable block. Emits none.
Tests: byte-identical output for the same inputs (a snapshot test); a changed part changes only what follows it.

### 3.2 Tool registry (`tools.py`)

```python
class ToolRegistry(Protocol):
    def register(
        self, 
        fn: Callable[..., Any], 
        *, 
        read_only: bool = False, 
        cacheable: bool = False,
        reads: Sequence[str] = (), 
        writes: Sequence[str] = ()) -> None: ...   # cacheable needs read_only
    def schemas(self) -> list[ToolSchema]: ...
    def execute(self, name: str, input: Mapping[str, Any]) -> ToolResult: ...
```

Plain typed functions become tools; the type hints are the only source of the JSON Schema (via `pydantic.TypeAdapter`) and the docstring is the description. `execute` is the one interface every "hand" shares and returns `ToolResult(ok, content)`. Invalid input and failures come back as `ok=False` with a short message that says how to fix the call, an observation the model can act on. `execute` is synchronous. Within one model turn the driver runs independent calls in threads; a tool may declare the resources it `reads` and `writes` (for example a path), and two calls conflict when either writes a resource the other reads or writes (read/write, write/read or write/write). A tool not registered `read_only=True` that declares an empty `writes` is treated as writing everything, whatever it declares in `reads`, so it conflicts with every other call in the turn. A conflict adds an edge from the earlier call to the later one in the order the model emitted them, so edges always point forward, the graph is acyclic by construction, and conflicting calls run in emission order. Read-only tools opt into an LRU cache keyed by `call_hash` plus a workspace generation counter, held by the `Workspace` itself (not derived from the log; `generation` and `bump()` are on its Protocol) and incremented on every write or delete. The registry is built with the `Workspace` it reads, and it calls `bump()` when any call that is not read-only returns, when such a call is abandoned on timeout, and again if an abandoned call later returns, which covers a write made by a sandboxed command. So a read after a write is never served stale. Emits none: it returns the result, and the driver appends `tool_completed` after the post guards have run, so a redacted result is never written raw. (How a refusal by the workspace or the sandbox is tied to the call that caused it, so that `sandbox_denied` carries its `tool_use_id`, is settled with the registry in card 07: those components only call the `on_refusal` they were built with.) Every `tool_use` (other than a block cut off by `max_tokens`, which is discarded) gets exactly one `tool_completed`. A call that a pre guard denies, that is declined at approval, that the sandbox refuses, or that does not run for any other reason is completed with `ok=false` and the reason as `content`, and the typed `finish` call is completed with `ok=true` and its result. For every call except `finish` this keeps the log equal to what the model saw (when a run ends on `finish`, the model never sees that result, and the log records it), and it is what makes a point after a denied call a clean fork point.
Tests: schemas for nested types; invalid input gives a useful error; independent calls overlap, conflicting calls run in emission order, and a non-`read_only` tool with an empty `writes` conflicts with all;  cache hits and eviction order; no caching for tools that write; a resumed run asks before re-running a tool that is not `read_only`; a timed-out call is abandoned and logged, and a late write still invalidates the cache.

### 3.3 Sandbox boundary (`sandbox.py`)

```python
class Sandbox(Protocol):
    def run(
        self, 
        cmd: Sequence[str], 
        *, 
        cwd: Path, 
        env_allow: Set[str], 
        limits: ResourceLimits) -> ProcessResult: ...

def evaluate(rules: Rules, request: Request) -> Permit | Refusal: ...   # pure; Refusal(action, target, reason) is what `sandbox_denied` logs
def docker_args(rules, request, limits, launch) -> list[str]: ...      # pure; the one `docker run` command line
```

The checks on the Docker client's own setup (config, context, `DOCKER_HOST`, proxies) live in `dockerenv.py`; `SandboxUnavailableError` is defined there.

Deny by default, and every command runs in its own Docker container. On macOS Docker Desktop runs the container inside a Linux virtual machine, so the command sits behind a hypervisor as well as the container; on Linux it shares the host kernel. There is no fallback: if the Docker executable, the daemon or the image is missing, `DockerSandbox` is not built (`SandboxUnavailableError`), and a sandbox never pulls an image on its own (`--pull never`), so what runs is what the owner pulled.

`Rules(root, image, commands, env, read)` says what is allowed. `commands` are absolute paths of executables *inside the image* (a bare name is refused, because searching a path would make the answer depend on the environment); `root` is the workspace (the working directory must resolve inside it); `env` is the ceiling on the variables a call may ask for; `read` are extra host trees mounted read-only. `Request(command, cwd, env_allow)` is checked in that order and the first rule that fails is the `Refusal`'s reason; the command is compared as written, so `/bin/ls/../rm` is not `/bin/rm`. An argument or working directory with a NUL byte or text that cannot be encoded, and a variable name that is not a name, are refusals too.

`DockerSandbox.run` resolves `cwd` (a relative one is relative to the workspace root, like the workspace's own paths), evaluates, and on a refusal calls its `on_refusal` callback (required: a sandbox whose refusals nobody hears would be a silent one) and raises `CommandRefusedError` carrying the `Refusal`; what the callback does with it is the caller's (card 07 turns it into `sandbox_denied`, §3.2). A permitted command runs as `docker run --rm` with a fixed command line that the caller cannot widen:

- **no network** (`--network none`), **no capabilities** (`--cap-drop ALL`), **no new privileges**, a **non-root user** (the caller's uid and gid, so what it writes into the workspace belongs to the owner; if loomcore itself runs as root the sandbox is not built, and `docker_args` rejects a root or malformed user), a **read-only root filesystem** with a 64 MB `noexec` `/tmp`, **no shared memory** (`--ipc none`: no `/dev/shm`), and an init process that reaps children, so the workspace and `/tmp` are the only writable places;
- the **workspace is the only writable mount**, at `/workspace`, which is also the working directory (a `cwd` below the root maps below `/workspace`); `Rules.read` trees are mounted read-only; a tree that contains the Docker daemon's socket (the default paths, `DOCKER_HOST`, and the endpoint of the Docker context in use, from `DOCKER_CONTEXT` or the config's `currentContext`, all of them when more than one is set; a context that cannot be found or is remote is an error unless `DOCKER_HOST` is set, which the CLI uses instead; the real one and where a link to it sits; when `DOCKER_HOST` is set, a context whose endpoint cannot be found (a missing, unreadable or malformed `meta.json`) or is not a local socket is ignored, and a readable local one is still refused as a tree) is refused as a workspace or a read tree, as is one at or under `/proc`, `/sys` or `/dev`, and a read tree that is, is under, or holds `/workspace`, because a read-only mount does not stop a command from connecting to a socket and the daemon is root on the host; the host's other files and processes are not there, and the container has its own process namespace, so the loomcore process and its environment cannot be read;
- **only the allowed variables** the parent has, plus what the image itself defines (`PATH` and the like), reach the command: they go in a private (mode 0600) `--env-file` that is deleted afterwards, never on a command line where a process listing would show them, and a value Docker's env-file cannot hold (a newline, carriage return or NUL, text that is not valid UTF-8, a line over 60 KB) is refused;
- **limits**: `timeout` (wall clock, then the container is removed by name with `docker rm --force`, which takes background children with it), `memory_bytes` (swap included), `pids`, `cpus`, `cpu_seconds` (CPU time of each process, not of the container), `max_file_bytes` (any one file), `max_open_files`, and `max_output_bytes` (stdout and stderr together, stdout first: past it the container is killed, and `output_truncated` is set whenever any output was dropped, even if the command had already exited). Nothing is left behind: when the call ends, times out or is interrupted, the container is removed by name with `docker rm --force`, repeated until the docker client exits, because `docker kill` cannot reach a container Docker has not started yet, and removed once more after the client is gone.

A command that cannot start (a program missing from the image, or a working directory that is missing or a file) is a `ProcessResult` with code 127, not an exception. `returncode` is Docker's: the command's own, 137 after a kill (the timeout always, and the output cap when Docker would have said 125 for the container it removed; otherwise, and for the memory limit, as Docker reports it), 125 if Docker itself failed. A limit that is not a finite number above 0 is a `ValueError` (Docker reads 0 and -1 as unlimited), and a Docker CLI config that sets `proxies`, or that exists but cannot be read as a JSON object, makes the sandbox unavailable (Docker would add those variables to the command's environment, and an unchecked config might). An empty `DOCKER_CONFIG` is unset, as the CLI treats it. An argument that is not strict UTF-8 (including the surrogate-escape range `U+DC80`–`U+DCFF`, which Docker would turn into `U+FFFD`) or holds a NUL byte is a refusal. The output cap bounds what is returned, not the host's temp-file use before the kill.

**Choosing the image.** The sandbox is distribution-agnostic: `Rules.image` is the owner's choice, and `commands` are paths inside it. The tests run in Ubuntu 24.04 LTS, pinned by digest (a tag can move, a digest cannot; CI pulls the same digest). Whatever the image, a container has no network and runs no SSH server, so a flaw in a daemon the image happens to ship is not reachable from outside, but the image is still the code the command runs on: prefer a small image (the Ubuntu base has about 90 packages, an Alpine one about 15, so fewer files for a flaw to hide in), a release with long security support, a digest in `Rules.image`, and a rebuild when an advisory names a package in it. Tools the tests use that are not in the base image (`wget`, `curl`) are not assumed.

**What the sandbox is not.** The workspace is mounted read-write on purpose: a command can change or delete anything in it, so keep what matters in version control. The container shares a kernel with the host on Linux and with the Docker virtual machine on macOS, so a kernel exploit is out of scope; the image is trusted, and so is whoever can reach the Docker daemon (access to it is equivalent to root on the host). Credentials the owner allow-lists in `env` are visible to the command, and the workspace itself is readable. Each command pays Docker's start-up cost, about a second. Isolation of the loomcore process itself is out of scope for v0.1, and so is a remote daemon (a `DOCKER_HOST`, or a context when `DOCKER_HOST` is not set, that is not a local `unix://` socket makes the sandbox unavailable, because a bind mount would name a path on the other machine; a remote context together with a local `DOCKER_HOST` builds, since the CLI uses `DOCKER_HOST`); `Rules.read` is trusted caller configuration, so a tree such as `~/.ssh` is readable by the command if the caller lists it; a rootless or Colima-style local socket is covered only through the context or `DOCKER_HOST` that points to it.
Tests: the rule and command-line tests need no Docker (no command outside the allow-list is ever permitted, a property test over `evaluate`; the command line has `--network none`, `--read-only`, `--cap-drop ALL`, a non-root user, and nothing that widens it; the workspace is the only writable mount), and each isolation property is shown by a real container from a small image (a parent secret is not passed, whether given as the environment or set in the real one, and the loomcore process's `/proc` entry is not there; host files are invisible; only the workspace is writable, and `/var/tmp` is read-only; the user is not root and every capability set is empty; there is no network, shown by an empty routing table and by every connection failing; there is no `/dev/shm`; the limits are the values the kernel was given (the container's cgroup shows the memory, swap and process caps); a timeout kills the container and its background children and leaves none, even when it arrives before Docker has created the container (the removals are made to miss in the test); the memory, process, CPU, file-size and output limits each bite). Mutants that drop a flag, run as root, pass the parent environment or skip a rule all fail the suite. In CI these tests fail, not skip, if Docker or the image is missing.

### 3.4 Filesystem workspace (`workspace.py`)

```python
class Workspace(Protocol):
    def read(self, path: str) -> str: ...
    def write(self, path: str, content: str) -> None: ...
    def list(self, path: str = ".") -> list[str]: ...
    def delete(self, path: str) -> None: ...
    @property
    def generation(self) -> int: ...        # incremented on every write or delete; the cache key uses it
    def bump(self) -> None: ...             # called by the registry for writes the workspace cannot see
```

The root is resolved once at construction (`root = root.resolve()`, so a symlinked root such as `/tmp` on macOS does not make every path fail). Every operation then goes through one `resolve_inside(root, path)`: resolve first (which follows symlinks), then require `is_relative_to(root)`. An absolute path is accepted only when it resolves inside the root (so `/etc/passwd` is refused and the root's own absolute path is not). A refusal of an operation raises `PathRefusedError`, which carries the `Refusal(action, target, reason)` that `sandbox_denied` logs; the workspace calls its `on_refusal` callback first, and what it does with it is the caller's (card 07 turns it into `sandbox_denied`, §3.2). Writing the notes file (a name given at construction, `notes.md` by default) calls `on_note`, which becomes `note_written`. `delete` removes the entry it names (a link is removed, not what it points to, even a link out or a dangling one), after checking that the entry's parent resolves inside the root; the root itself cannot be deleted. `list` names the entries as the path asked for them, so through a link to a directory it returns `link/x`, and for an absolute path it returns absolute names. A path that cannot be resolved (a NUL byte) is a refusal and is logged. The notes file must be inside the root (a name that is not is a configuration error raised at construction, before any tool call, and is not reported through `on_refusal`), and a write counts as a note write when it is that file (the same file, so another spelling or a link to it counts). `write` creates missing parents inside the root, and the generation is bumped even by a write that fails, because it may have changed the file. The check is not atomic with the open that follows, so a process that can race the workspace can swap a symlink in between; the sandbox, not the workspace, contains such a process.
Tests, written first: `../`, absolute paths, a symlink out of the workspace, a symlinked parent directory, a dangling link out, a symlinked root; a property test that no read, write, delete or list on a path built from `..`, `.`, names, links and slashes reads or changes a file outside the root (the oracle is the disk, and the test fails if the containment check is removed).

### 3.5 Memory and context store (`context.py`)

```python
class ContextStore(Protocol):
    def pack(
        self, 
        candidates: Sequence[Item], 
        budget: int, 
        pinned: Sequence[Item]) -> Packed: ...
    def compact(self, packed: Packed) -> Packed: ...

class Formatter(Protocol):
    def render(self, events: Sequence[Event]) -> list[Message]: ...
```

`pack` is pure: each `Item` carries its token count, measured at the edge, so `pack` does no counting itself. Pinned items (system prompt, tool definitions, the latest user turn and tool results) are always in; the rest are chosen greedily by value per token with a heap, ties broken deterministically. Compaction runs at a threshold (for example 80% of the budget): clear old tool results first, then summarise. A notes file in the workspace holds what the agent wants to keep outside the window. Resolved errors and failed calls are hidden from later windows. The formatter is swappable, so the same log can render as messages or as tagged blocks. Tokens are counted, not estimated. Emits `context_packed` and `context_compacted`.
Tests: packing respects the budget and always includes pinned items; compaction fires at the threshold; the stable prefix is identical across turns.

### 3.6 Feedback and self-verification hook (`verify.py`)

```python
Verifier = Callable[[RunState], VerifyResult]
```

When the model claims it is done (a `finish` call or an `end_turn`), the verifier runs. A failure becomes a `verification_failed` observation with a short critique and the loop continues; a pass lets the run end. Verifiers check the outcome (tests, a checker function), not only the code. Emits `verification_passed` and `verification_failed`.
Tests: a failing verifier prevents `finished` and feeds back; a passing one ends the run.

### 3.7 Guardrails and the human-in-the-loop seam (`guards.py`)

```python
class Guard(Protocol):                                    # stage "pre": before the tool runs
    def check(
        self, 
        request: ToolRequest, 
        state: RunState) -> Allow | Deny | Ask: ...

class PostGuard(Protocol):                                # stage "post": on the result, before it is logged
    def check(
        self, 
        request: ToolRequest, 
        result: ToolResult, 
        state: RunState) -> Allow | Deny | Redact: ...
```

The decisions are frozen dataclasses with a defined payload:

```python
@dataclass(frozen=True)
class Allow: ...

@dataclass(frozen=True)
class Deny:
    reason: str                  # logged on guard_decision; for a pre guard, also the observation the model sees

@dataclass(frozen=True)
class Ask:
    question: str                # logged on approval_requested

@dataclass(frozen=True)
class Decision:                  # what `resume` receives for an approval; logged on approval_granted / approval_denied
    approved: bool
    by: str
    note: str = ""

@dataclass(frozen=True)
class Redact:
    content: str                 # replaces the result's content, in the log and for the model
    reason: str                  # what was removed and why, with no secret in it; logged on guard_decision
```

The driver runs both pipelines when it performs a `RunTool` effect: the pre guards before the tool, the post guards on its result. A denial's completion is that effect's result, so `step` stays pure and nothing in it needs to know where the guards run. `Redact` is only valid from a post guard, and it continues the pipeline with the replaced content. The driver logs `content`, never the original, and the original is dropped. This is the one deliberate exception to "nothing hidden" in the intent: the log holds the redacted content, not the secret. If several post guards redact in turn, each sees the previous one's output. A post `Deny` replaces the result with an `ok=False` observation that says why. A post guard cannot `Ask`, because the tool has already run. Guards form two pipelines of pure checks run in order, stopping at the first `Deny` or `Ask` (`pre`, before a tool runs) or the first `Deny` (`post`, on its output: for example a secret in a result, which `Redact` removes, or an oversized one). Tool results are untrusted data, wrapped and marked as data in the prompt. `Ask` appends `approval_requested`, the run pauses (`run_paused`, `awaiting_approval`), and `resume(ref, decision)` appends `approval_granted` or `approval_denied` and continues from the log. The pause sits between choosing a tool and running it, so nothing executes before approval. Budget enforcement is not a guard: it is the pure `check_limits` over the state in section 1. Emits `guard_decision` and the approval events.
Tests: a deny stops the tool and is logged; a post guard redacts a secret and the raw result never reaches the log or the model; an ask pauses the run; kill the process, resume, and the run continues; `check_limits` stops at the cap boundary with exact decimal arithmetic, counts a run's own spend, and prices the worst case.

### 3.8 Observability (`observability/`)

```python
def spans(events: Iterable[Event]) -> list[Span]: ...             # trace.py
def price(usage: Usage, table: PricingTable) -> Decimal: ...      # cost.py
```

Spans are derived from the event log and never written separately: a run is a trace, and each turn, model call and tool call is a span with timing and attributes, using the OpenTelemetry `gen_ai.*` attribute names where they exist. Durations come from the `duration_ms` logged on `model_call` and `tool_completed`, measured with the `Monotonic` port when the call is made; spans fall back to timestamp differences where it is missing. `cost.py` tallies tokens per call and per run and prices them from `pricing.json`. Emits no events of its own.
Tests: spans from a recorded log are stable; pricing matches hand-computed values; cache read and cache write tokens are priced separately.

## 4. Layout

```
src/loomcore/
  events.py      # envelope and kinds
  eventlog.py    # writer, reader
  replay.py      # run state, fold, a run's events across forks
  snapshot.py    # snapshots
  rewind.py      # clean fork points
  spend.py       # spend log
  step.py        # step and its effects
  retry.py       # retry policy
  months.py      # the UTC month both spend and the run state use
  limits.py      # Limits, check_limits, limits in force
  ports.py       # ports and the driver's value types
  conversation.py # the messages a call sends, rebuilt from the log
  engine.py      # one run: emit events, perform effects
  loop.py        # Driver: run, resume, fork
  ladder.py      # the repeat detector and the rung messages
  finish.py      # the typed finish tool
  tools.py       # registry
  sandbox.py     # sandbox and rule evaluation
  dockerenv.py   # what the Docker client is set up to do: its socket, its proxies
  workspace.py   # filesystem workspace
  context.py     # pack, compaction, formatter
  prompt.py      # system-prompt builder
  guards.py      # guard pipeline
  verify.py      # verifier hook
  observability/
    trace.py
    cost.py
tests/
evals/           # the scripted three-step task
```

## 5. Tests and the eval

Unit tests sit next to the behaviour they prove. The event log also gets [Hypothesis](https://hypothesis.readthedocs.io/en/latest/stateful.html) stateful tests (random appends, snapshots and replays always agree with the fold) and a truncation test (cut the file at a random byte; replay still gives a consistent earlier state), plus truncate, resume, append and read again; a fork counts each model call once; crash, wait, `resume`: the run does not time out and `run_resumed` marks the gap; a limit stop pauses, `resume` after a limit pause without `limits` is an error, `resume` with limits continues with the turn count, spend and clock carried over and pauses again with the same `limit` if it is still reached, `resume` on a finished run is an error, and a sibling fork's limits never apply to the parent; a fork at a point with a pending tool call (including a `model_call` whose `tool_use` has no `tool_requested` yet, or one of several calls done), an outstanding approval (including the ladder's, before its `run_paused`) or an open pause is refused; `max_tokens` twice in a row pauses with `max_output_tokens`, one other call between them resets the count, and so do a `run_resumed` after a `max_output_tokens` pause and a `run_forked` (a fork taken after two truncations does not pause on its first check), while a crash resume mid-streak still pauses; a denied, declined or refused call still gets one `tool_completed` with `ok=false`, and a fork after it is clean; identical calls trip the detector at `repeat_threshold` within `repeat_window` and not before; a deliberate repeat is never refused by nudge or switch, which add their message to the result; two trips in one turn climb one rung; a fork starts at rung 0 and counts only its own calls; asking a human holds the whole turn until the decision, a denial completes only the asked call, and after a grant the next trip goes to `stuck`, and the next identical call does not trip until `repeat_threshold` identical calls have been made since the grant; a post-guard `Redact` keeps the rung's message; `stuck` completes every pending call in the turn before `run_finished`; a varying-argument loop is not detected and ends at `max_turns`; a fork after any rung is clean once the completions are written; a fork starts its own counts but `month_cap` still binds, and a run that spans a month boundary charges only the current month's spend to `month_cap`; a fork inherits the parent's `run_resumed.limits` written at or before `from.seq`, one written after does not apply to it, and a later `run_resumed` without `limits` does not undo an override; a kill or error with calls in flight completes them `ok=false` before `run_finished`; a `tool_use` cut off by `max_tokens` is discarded and never run; a `delete` invalidates the cache like a write; a `sandbox_denied` caused by a tool call carries its `tool_use_id`; `resume` refuses a changed digest, including after a switch-strategy configuration; `finish` is completed `ok=true` with its result, and a completion for a call that did not run has `duration_ms` 0 and `cached` false; a non-`read_only` tool that declares only `reads` conflicts with every other call; a crash during a kill or error settles pending calls as after any crash; two writers on one session file are refused, and a crash while holding the lock does not block `resume`; a torn spend-log append does not swallow the next run's line; `resume` after a crash recomputes the missing spend line; a second run's spend counts against `month_cap`; and a fork's clock and turn count start at its `run_forked`.

The eval is a scripted three-step task driven by a fake model that replays canned responses. It runs in CI like a test and must reach `finished` with a complete, replayable log.

## 6. Done for v0.1

A second repository can `import loomcore`, register a tool and run a multi-step task to completion, with the full event log persisted and replayable. Every component has its `Protocol` and tests, the scripted eval is green in CI, and the cost module prices a recorded log from tokens alone.
