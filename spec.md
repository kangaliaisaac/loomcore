# `loomcore`: spec

- [`loomcore`: spec](#loomcore-spec)
  - [1. The loop](#1-the-loop)
    - [Ending a run](#ending-a-run)
    - [Limits](#limits)
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
def step(state: RunState, event: Event) -> tuple[RunState, list[Effect]]: ...
```

1. `step` never performs I/O. A thin driver feeds it events, performs the effects it returns (call the model, run a tool, ask a human, stop) and appends each result to the log as the next event. 
2. `RunState` is a fold over the log: `functools.reduce(apply, events, initial)`, with `apply` pure. Given the same configuration, a new process can load the log and continue: the configuration (system prompt, tool schemas, limits) is logged once in `run_started`, and `resume` refuses to continue if its digest differs.

```python
# Effect = CallModel | RunTool | AskHuman | Stop

class Driver(Protocol):
    def run(self, task: Task, log: EventLog) -> RunResult: ...
    def resume(self, ref: RunRef, decision: Decision | None = None) -> RunResult: ...   # RunRef = (session_id, run_id)
```

Every component below works without the driver: a user may call the registry, the sandbox, the guards and the context store from their own loop.

**Ports** (the only places that touch the world, injected into the driver):

```python
ModelCall = Callable[[ModelRequest], ModelResponse]       # wraps the Anthropic SDK; tests use a scripted fake
Clock = Callable[[], datetime]                            # wall-clock UTC
Monotonic = Callable[[], float]                           # durations
Random = Callable[[], float]                              # jitter; tests pass a fixed one
```

### Ending a run

A run ends on purpose. `run_finished` is terminal and carries the reason; a run that waits for a human writes `run_paused` instead and is later resumed, so a run has at most one `run_finished`.

| Event | Reason | Meaning |
|---|---|---|
| `run_finished` | `finished` | the model called the typed `finish(result)` tool and verification passed |
| `run_finished` | `end_turn` | the model stopped without `finish` and no tool was pending; the verifier runs here too, and a failure continues the loop |
| `run_finished` | `max_turns`, `timeout`, `budget`, `max_tokens` | a harness limit was reached; the stop is graceful and records what was done and the `seq` to continue from. The run is finished: to continue, fork from that `seq` with raised limits (rewind), and `resume` on a finished run is an error |
| `run_finished` | `refused` | the model's `stop_reason` was `refusal` |
| `run_finished` | `stuck` | the escalation ladder ran out |
| `run_finished` | `killed` | an external signal or the kill switch; a budget cap gives `budget`, not `killed` |
| `run_finished` | `error` | an unrecoverable error, with the message |
| `run_paused` | `awaiting_approval` | a guard, the escalation ladder, or crash resume of a tool that is not read-only asked for approval; `resume` with a decision appends `run_resumed` and continues the same run |

Other model stop reasons are handled and logged: 
- `tool_use` continues the loop, 
- `stop_sequence` is treated as `end_turn`, 
- `max_tokens` continues once and then stops with the reason `max_tokens`, and 
- `pause_turn` resumes by sending the response back.

### Limits

Turn, time and budget limits are part of the core, not extensions. `max_turns` and `timeout` have defaults (50 turns, 10 minutes). The budget has two caps, `run_cap` (this run) and `month_cap` (all runs this UTC month); neither has a default, and each must be given or explicitly set to `None` for unlimited. The check stops with `budget` when either would be crossed. Before each model call the driver checks the limits as a pure function over the state: `check_limits(state, limits, month_to_date, worst_case, now) → Allow | Stop(reason)`.

- **Per-run spend** is a fold over the `model_call` events in the state whose `runId` equals this run. A forked run inherits its parent's events for context but not for spend.
- **`month_to_date`** is read from the spend log at the edge and passed in: the sum of entries whose `at` falls in the current UTC month, excluding this run's own lines. The check compares per-run spend plus `worst_case` with `run_cap`, and `month_to_date` plus per-run spend plus `worst_case` with `month_cap`, so nothing is counted twice across a pause.
- **`worst_case`** prices the next call at its `max_tokens` ceiling, which every `ModelRequest` carries.
- **`timeout`** counts active time, as a sum of segments. A segment starts at `run_started`, `run_forked` or `run_resumed` and ends at the last event before the next gap (a `run_paused`, or the end of the log after a crash); the open segment ends at `now` (the `Clock` port). A run waiting an hour for approval, or resumed hours after a crash, does not time out for the time it was down.

**Retries** are a pure policy, `(attempt, error, random) → Delay | GiveUp`, with the random source passed in. 429 and 529, other 5xx, connection errors and timeouts get exponential backoff with full jitter (`random() * min(cap, base * 2**attempt)`) and honour `retry-after`; any other 4xx is never retried. Every model and tool call has a timeout. A tool timeout abandons the call: it is logged as failed with `timeout` and the loop moves on, but Python cannot kill a thread, so the call may keep running. A tool that writes must therefore run through the sandbox (a subprocess, killed on timeout) or accept a cancel event; a late write from an abandoned call still bumps the workspace generation counter, so the cache stays correct.

**Two identifiers, two jobs.** A replay or a retry does not re-run a completed call: tool completions are matched by `tool_use_id`, and if the log already holds a `tool_completed` for it, the result is reused. Delivery is at-least-once. If the process dies after a tool runs but before its completion is appended, `resume` re-runs a tool registered `read_only=True`, and turns a call to any other tool, declared or not, into an `Ask` before running it again. A tool that does not say it is read-only is treated as one that has side effects. Separately, `call_hash` is `sha256(json.dumps({"tool": name, "input": args}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()`, stored on `tool_requested`; it identifies a repeated call for the repeat detector and the cache, and a deliberate repeat (running the tests again after an edit) is still allowed.

### Escalation ladder

The repeat detector counts the same `call_hash` in a run. Past a threshold the ladder climbs one rung at a time: nudge (a short corrective message), switch strategy (a different prompt or tool set), ask a human (an `Ask`), stop with the best partial result (`stuck`). Each rung is an event.

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

**Rewind** forks a run: a new `run_id` in the same session file starts with a `run_forked` event holding `from: {runId, seq}` and optionally a human hint, then continues. No events are copied, so replaying the new run reads the old run's events up to `seq` and then its own, and every model call is counted once. The old run is untouched. A fork has no `run_started` of its own: its configuration is its root ancestor's, taken from the first `run_started` up the `from` chain (a fork of a fork follows the chain to the root), and `resume` compares that digest. Replay follows `from` recursively. The repeat detector counts only the fork's own calls, like spend, so a fork gets a fresh start from that point. Its `timeout` clock starts at its `run_forked` timestamp and its turn count starts at 0, so neither includes the parent's time or turns, and its per-run spend counts only its own `model_call` events. Rewind restores conversation state, not files: what the old run wrote after `seq` is still on disk, and the fork starts from that workspace.

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
| `run_started` | driver | `task`, `config` (system prompt, tool schemas, limits), `config_digest` |
| `run_forked` | driver | `from` (`runId`, `seq`), `hint` |
| `turn_started` | driver | `turn` |
| `model_call` | driver | `type: "assistant"`, `message`, `duration_ms`, `cost_usd`, `pricing_version` (below) |
| `tool_requested` | driver | `tool_use_id`, `name`, `input`, `call_hash` |
| `tool_completed` | driver | `tool_use_id`, `ok`, `content`, `duration_ms`, `cached` |
| `guard_decision` | guards | `stage` (`pre`/`post`), `tool_use_id`, `decision` (`allow`/`deny`/`ask`/`redact`; `ask` only at `pre`, `redact` only at `post`), `reason` |
| `approval_requested` | guards, driver | `tool_use_id` (null when the escalation ladder asks), `question` |
| `approval_granted`, `approval_denied` | driver | `tool_use_id` (null likewise), `by`, `note` |
| `sandbox_denied` | sandbox, workspace | `action`, `target`, `reason` |
| `context_packed` | context store | `selected`, `dropped`, `tokens`, `budget` |
| `context_compacted` | context store | `strategy`, `tokens_before`, `tokens_after` |
| `verification_passed`, `verification_failed` | verifier | `verifier`, `message` |
| `escalation` | driver | `rung`, `reason` |
| `note_written` | workspace | `path` |
| `run_paused` | driver | `reason`, `resume_from` |
| `run_resumed` | driver | `reason` (`approval` or `crash`), `resume_from` |
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

- The writer appends one line and flushes; an `fsync` option exists for durability-sensitive runs. A session file has one writer at a time: the writer holds an exclusive lock file (`<sessionId>.lock`), and a second process that wants to write waits or fails, so a fork cannot run while its parent is resumed. Before its first append to an existing file the writer, holding the lock, cuts the file back to the last complete newline, so a torn final line from a crash never ends up in the middle of the log.
- An optional redactor runs on each event before it is appended, so a secret that reached a tool result can be removed from the log.
- The reader yields events in order. A broken *final* line (a torn write) is dropped and reported; a broken line anywhere else is an error.
- Snapshots of derived state are written to a temporary file and moved with `os.replace`, so they are atomic. Replay may start from a snapshot.
- Replay is deterministic: the same log always yields the same state.

### Tokens and cost

Token counts are the source of truth and cost is derived. `cost_usd` on an event is advisory; reports re-price from tokens with the versioned table named by `pricing_version`. The table is a vendored `pricing.json`. Money is `decimal.Decimal`.

### Run spend line

After each `model_call`, `loomcore` appends one line to a spend log: `{"at", "usd", "run", "note"}`, where `run` is `"<sessionId>/<runId>"` and `usd` is the cost of that call, counting only events whose `runId` is that run. Writing per call, not at the end, means a crashed run and a run still in progress are both already in the log, so the month cap sees them. One line is a single small append, so concurrent runs can share the file without the session lock. Each line stands alone, so the reader skips a line that fails to parse (a crash mid-append) and reports it, as it does for a torn final line in an event log. `month_to_date` for `check_limits` is read from this log as defined in section 1: the current UTC month, excluding the run's own lines. If a process dies between a `model_call` and its spend line, `resume` sums the run's `model_call` events, subtracts the run's lines, and writes the difference first. Concurrent runs can overshoot a cap by at most one call each, because the worst case is priced before every call but another run may spend in the meantime.

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
    def register(self, fn: Callable[..., Any], *, read_only: bool = False, cacheable: bool = False,
                 reads: Sequence[str] = (), writes: Sequence[str] = ()) -> None: ...   # cacheable needs read_only
    def schemas(self) -> list[ToolSchema]: ...
    def execute(self, name: str, input: Mapping[str, Any]) -> ToolResult: ...
```

Plain typed functions become tools; the type hints are the only source of the JSON Schema (via `pydantic.TypeAdapter`) and the docstring is the description. `execute` is the one interface every "hand" shares and returns `ToolResult(ok, content)`. Invalid input and failures come back as `ok=False` with a short message that says how to fix the call, an observation the model can act on. `execute` is synchronous. Within one model turn the driver runs independent calls in threads; a tool may declare the resources it `reads` and `writes` (for example a path), and two calls conflict when either writes a resource the other reads or writes (read/write, write/read or write/write). A tool not registered `read_only=True` that declares no resources is treated as writing everything, so it conflicts with every other call in the turn. A conflict adds an edge from the earlier call to the later one in the order the model emitted them, so edges always point forward, the graph is acyclic by construction, and conflicting calls run in emission order. Read-only tools opt into an LRU cache keyed by `call_hash` plus a workspace generation counter, held by the `Workspace` itself (not derived from the log; `generation` and `bump()` are on its Protocol) and incremented on every write. The registry is built with the `Workspace` it reads, and it calls `bump()` when any call that is not read-only returns, when such a call is abandoned on timeout, and again if an abandoned call later returns, which covers a write made by a sandboxed command. So a read after a write is never served stale. Emits none: it returns the result, and the driver appends `tool_completed` after the post guards have run, so a redacted result is never written raw.
Tests: schemas for nested types; invalid input gives a useful error; independent calls overlap, conflicting calls run in emission order, and a tool with no declared resources conflicts with all;  cache hits and eviction order; no caching for tools that write; a resumed run asks before re-running a tool that is not `read_only`; a timed-out call is abandoned and logged, and a late write still invalidates the cache.

### 3.3 Sandbox boundary (`sandbox.py`)

```python
class Sandbox(Protocol):
    def run(self, cmd: Sequence[str], *, cwd: Path, env_allow: Set[str], limits: ResourceLimits) -> ProcessResult: ...

def evaluate(rules: Rules, request: Request) -> Allow | Deny: ...   # pure
```

Deny by default. Commands run with `shell=False`, a timeout and resource limits, and only the environment variables on an explicit allow-list; the parent environment is never inherited, so no credential is reachable from code the model can run unless the user explicitly allow-lists it. Rule evaluation is a pure function. A denial emits `sandbox_denied`.
Tests: a secret set in the parent process does not appear in the child; a timeout kills the process; every denial is logged.

### 3.4 Filesystem workspace (`workspace.py`)

```python
class Workspace(Protocol):
    def read(self, path: str) -> str: ...
    def write(self, path: str, content: str) -> None: ...
    def list(self, path: str = ".") -> list[str]: ...
    def delete(self, path: str) -> None: ...
    @property
    def generation(self) -> int: ...        # incremented on every write; the cache key uses it
    def bump(self) -> None: ...             # called by the registry for writes the workspace cannot see
```

The root is resolved once at construction (`root = root.resolve()`, so a symlinked root such as `/tmp` on macOS does not make every path fail). Every operation then goes through one `resolve_inside(root, path)`: resolve first (which follows symlinks), then require `is_relative_to(root)`. Emits `sandbox_denied` on an escape and `note_written` for notes.
Tests, written first: `../`, absolute paths, a symlink out of the workspace, a symlinked parent directory.

### 3.5 Memory and context store (`context.py`)

```python
class ContextStore(Protocol):
    def pack(self, candidates: Sequence[Item], budget: int, pinned: Sequence[Item]) -> Packed: ...
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
    def check(self, request: ToolRequest, state: RunState) -> Allow | Deny | Ask: ...

class PostGuard(Protocol):                                # stage "post": on the result, before it is logged
    def check(self, request: ToolRequest, result: ToolResult, state: RunState) -> Allow | Deny | Redact: ...
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
class Redact:
    content: str                 # replaces the result's content, in the log and for the model
    reason: str                  # what was removed and why, with no secret in it; logged on guard_decision
```

`Redact` is only valid from a post guard, and it continues the pipeline with the replaced content. The driver logs `content`, never the original, and the original is dropped. This is the one deliberate exception to "nothing hidden" in the intent: the log holds the redacted content, not the secret. If several post guards redact in turn, each sees the previous one's output. A post `Deny` replaces the result with an `ok=False` observation that says why. A post guard cannot `Ask`, because the tool has already run. Guards form two pipelines of pure checks run in order, stopping at the first `Deny` or `Ask` (`pre`, before a tool runs) or the first `Deny` (`post`, on its output: for example a secret in a result, which `Redact` removes, or an oversized one). Tool results are untrusted data, wrapped and marked as data in the prompt. `Ask` appends `approval_requested`, the run pauses (`run_paused`, `awaiting_approval`), and `resume(ref, decision)` appends `approval_granted` or `approval_denied` and continues from the log. The pause sits between choosing a tool and running it, so nothing executes before approval. Budget enforcement is not a guard: it is the pure `check_limits` over the state in section 1. Emits `guard_decision` and the approval events.
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
  events.py      # envelope, kinds, writer, reader, snapshot, replay
  loop.py        # step, Driver, ports, retry policy, limits, escalation
  tools.py       # registry
  sandbox.py     # sandbox and rule evaluation
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

Unit tests sit next to the behaviour they prove. The event log also gets [Hypothesis](https://hypothesis.readthedocs.io/en/latest/stateful.html) stateful tests (random appends, snapshots and replays always agree with the fold) and a truncation test (cut the file at a random byte; replay still gives a consistent earlier state), plus truncate, resume, append and read again; a fork counts each model call once; crash, wait, `resume`: the run does not time out and `run_resumed` marks the gap; `resume` on a finished run is an error and a limit stop continues by fork; two writers on one session file are refused; `resume` after a crash recomputes the missing spend line; a second run's spend counts against `month_cap`; and a fork's clock and turn count start at its `run_forked`.

The eval is a scripted three-step task driven by a fake model that replays canned responses. It runs in CI like a test and must reach `finished` with a complete, replayable log.

## 6. Done for v0.1

A second repository can `import loomcore`, register a tool and run a multi-step task to completion, with the full event log persisted and replayable. Every component has its `Protocol` and tests, the scripted eval is green in CI, and the cost module prices a recorded log from tokens alone.
