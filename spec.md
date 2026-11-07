# loomcore: spec

The contract for v0.1. [intent.md](intent.md) says what loomcore is for; this file says what it does, component by component. Where the two differ, this file wins. Python 3.15, `src/` layout, full type hints, mypy strict. Interfaces are `Protocol`s; data is frozen dataclasses; effects sit at the edge and are passed in.

## 1. The loop

loomcore runs a ReAct-style loop over native tool use: the model emits `tool_use` blocks, the harness runs them and returns `tool_result` blocks, and `stop_reason` says when the turn is over. The paper's parsed "Thought / Action / Observation" text is not used; extended thinking and tool blocks replace it.

The core is one pure function:

```python
def step(state: RunState, event: Event) -> tuple[RunState, list[Effect]]: ...
```

`step` never performs I/O. A thin driver feeds it events, performs the effects it returns (call the model, run a tool, ask a human, stop) and appends each result to the log as the next event. `RunState` is a fold over the log: `functools.reduce(apply, events, initial)`, with `apply` pure. Nothing about a run lives outside the log, so a new process can load the log and continue.

```python
Effect = CallModel | RunTool | AskHuman | Stop

class Driver(Protocol):
    def run(self, task: Task, log: EventLog) -> RunResult: ...
    def resume(self, run_id: str, decision: Decision | None = None) -> RunResult: ...
```

Every component below works without the driver: a user may call the registry, the sandbox, the guards and the context store from their own loop.

**Ports** (the only places that touch the world, injected into the driver):

```python
ModelCall = Callable[[ModelRequest], ModelResponse]       # wraps the Anthropic SDK; tests use a scripted fake
Clock = Callable[[], datetime]                            # wall-clock UTC
Monotonic = Callable[[], float]                           # durations
```

### Ending a run

A run ends on purpose, and the reason is logged as `run_finished.reason`:

| Reason | Meaning |
|---|---|
| `finished` | the model called the typed `finish(result)` tool |
| `end_turn` | the model stopped without calling `finish` and no tool was pending |
| `max_turns`, `budget`, `timeout` | a harness limit; the stop is graceful and records what was done and where to resume |
| `awaiting_approval` | a guard returned `Ask`; resume with a decision |
| `stuck` | the escalation ladder ran out |
| `killed` | the kill switch (a budget cap or an external signal) |
| `error` | an unrecoverable error, with the message |

Model stop reasons are handled and logged: `end_turn`, `tool_use`, `max_tokens` (continue or stop by policy), `stop_sequence`, `pause_turn` (resume by sending the response back) and `refusal`.

**Retries** are a pure policy, `(attempt, error) → Delay | GiveUp`. 429 and 529 get exponential backoff with full jitter (`uniform(0, min(cap, base * 2**attempt))`) and honour `retry-after`; request errors (400-class) are never retried. Every model and tool call has a timeout.

**Idempotency key.** `sha256(json.dumps({"tool": name, "input": args}, sort_keys=True, separators=(",", ":")))`. It is stored on `tool_requested`, so a retry or a replay never runs a side effect twice.

### Escalation ladder

The repeat detector counts the same idempotency key in a run. Past a threshold the ladder climbs one rung at a time: nudge (a short corrective message), switch strategy (a different prompt or tool set), ask a human (an `Ask`), stop with the best partial result (`stuck`). Each rung is an event.

### Extension seams

An extension plugs into a seam that already exists and never changes the core. Each is off by default and enabled by configuration.

| Seam | Examples |
|---|---|
| A tool | the plan tool (a plan file always kept in context), `finish`, note-taking |
| A hook around `step` | verify then reflect, a periodic critique, replan on failure |
| A stop policy | turn, budget and time limits, the escalation ladder |
| An alternative driver reusing the core | bring your own loop |

**Built in v0.1:** the escalation ladder, `finish` and graceful stop, the plan tool, verify then reflect, rewind, bring-your-own-loop. **Named interfaces only, not built:** `PlannerExecutor`, `ScriptAction`, `ModelRouter`, `SubAgent`, `BestOfN`, `TreeSearch`. Each is a `Protocol` with a docstring and no implementation.

**Rewind** forks a run: a new `run_id` whose log is the first *k* events of the old one, optionally followed by a human hint, then continues. The old log is untouched.

## 2. The event log

The log is append-only JSONL, one event per line, and the only record of a run. Everything that reports on a run reads it; nothing reads another reader's output.

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

The field names `sessionId`, `timestamp` and the `message` block below follow the transcript format that existing Claude Code tooling already parses, so a loomcore log needs a new root directory and no new parser. Files live at `<runs-root>/<project>/<sessionId>.jsonl`, with sub-run logs under `subagents/`.

### Kinds

| Kind | Emitted by | Payload (besides the envelope) |
|---|---|---|
| `run_started` | driver | `task`, `config_digest` |
| `turn_started` | driver | `turn` |
| `model_call` | driver | `type: "assistant"`, `message`, `cost_usd`, `pricing_version` (below) |
| `tool_requested` | driver | `tool_use_id`, `name`, `input`, `idempotency_key` |
| `tool_completed` | registry | `tool_use_id`, `ok`, `content`, `duration_ms`, `cached` |
| `guard_decision` | guards | `stage` (`pre`/`post`), `tool_use_id`, `decision` (`allow`/`deny`/`ask`), `reason` |
| `approval_requested` | guards | `tool_use_id`, `question` |
| `approval_granted`, `approval_denied` | driver | `tool_use_id`, `by`, `note` |
| `sandbox_denied` | sandbox, workspace | `action`, `target`, `reason` |
| `context_packed` | context store | `selected`, `dropped`, `tokens`, `budget` |
| `context_compacted` | context store | `strategy`, `tokens_before`, `tokens_after` |
| `verification_passed`, `verification_failed` | verifier | `verifier`, `message` |
| `escalation` | driver | `rung`, `reason` |
| `note_written` | workspace | `path` |
| `run_finished` | driver | `reason`, `result`, `resume_from` |

The `model_call` line has the same `usage` block the Anthropic Messages API returns:

```json
{"schema": "loomcore.event.v1", "seq": 7, "kind": "model_call", "type": "assistant",
 "timestamp": "2026-11-03T08:14:02.311Z", "sessionId": "run-2026-11-03-0814", "runId": "r1",
 "message": {"id": "msg_01…", "model": "claude-opus-5-5",
   "usage": {"input_tokens": 1840, "output_tokens": 312,
     "cache_read_input_tokens": 9200, "cache_creation_input_tokens": 0,
     "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}},
   "content": [{"type": "tool_use", "name": "retrieve_passages"}]},
 "cost_usd": 0.0129, "pricing_version": "2026-10"}
```

### Writing, reading, replay

- The writer appends one line and flushes; an `fsync` option exists for durability-sensitive runs.
- The reader yields events in order. A broken *final* line (a torn write) is dropped and reported; a broken line anywhere else is an error.
- Snapshots of derived state are written to a temporary file and moved with `os.replace`, so they are atomic. Replay may start from a snapshot.
- Replay is deterministic: the same log always yields the same state.

### Tokens and cost

Token counts are the source of truth and cost is derived. `cost_usd` on an event is advisory; reports re-price from tokens with the versioned table named by `pricing_version`. The table is a vendored `pricing.json`. Money is `decimal.Decimal`.

### Run spend line

When a run ends, loomcore appends one line to a spend log: `{"at", "usd", "run", "note"}`. The budget guard reads this log.

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
    def register(self, fn: Callable[..., Any], *, cacheable: bool = False) -> None: ...
    def schemas(self) -> list[ToolSchema]: ...
    def execute(self, name: str, input: Mapping[str, Any]) -> ToolResult: ...
```

Plain typed functions become tools; the type hints are the only source of the JSON Schema (via `pydantic.TypeAdapter`) and the docstring is the description. `execute` is the one interface every "hand" shares and returns `ToolResult(ok, content)`. Invalid input and failures come back as `ok=False` with a short message that says how to fix the call, an observation the model can act on. Independent calls run concurrently (`asyncio.TaskGroup`); dependent calls are ordered by `graphlib.TopologicalSorter`, and a cycle becomes a clean error. Read-only tools opt into an LRU cache keyed by the idempotency key. Emits `tool_completed`.
Tests: schemas for nested types; invalid input gives a useful error; independent calls overlap, dependent calls run in order; a cycle is reported; cache hits and eviction order; no caching for tools that write.

### 3.3 Sandbox boundary (`sandbox.py`)

```python
class Sandbox(Protocol):
    def run(self, cmd: Sequence[str], *, cwd: Path, env_allow: Set[str], limits: Limits) -> RunResult: ...

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
```

Every operation goes through one `resolve_inside(root, path)`: resolve first (which follows symlinks), then require `is_relative_to(root)`. Emits `sandbox_denied` on an escape and `note_written` for notes.
Tests, written first: `../`, absolute paths, a symlink out of the workspace, a symlinked parent directory.

### 3.5 Memory and context store (`context.py`)

```python
class ContextStore(Protocol):
    def pack(self, candidates: Sequence[Item], budget: int, pinned: Sequence[Item]) -> Packed: ...
    def compact(self, packed: Packed) -> Packed: ...

class Formatter(Protocol):
    def render(self, events: Sequence[Event]) -> list[Message]: ...
```

`pack` is pure: pinned items (system prompt, tool definitions, the latest user turn and tool results) are always in; the rest are chosen greedily by value per token with a heap, ties broken deterministically. Compaction runs at a threshold (for example 80% of the budget): clear old tool results first, then summarise. A notes file in the workspace holds what the agent wants to keep outside the window. Resolved errors and failed calls are hidden from later windows. The formatter is swappable, so the same log can render as messages or as tagged blocks. Tokens are counted, not estimated. Emits `context_packed` and `context_compacted`.
Tests: packing respects the budget and always includes pinned items; compaction fires at the threshold; the stable prefix is identical across turns.

### 3.6 Feedback and self-verification hook (`verify.py`)

```python
Verifier = Callable[[RunState], VerifyResult]
```

When the model claims it is done, the verifier runs. A failure becomes a `verification_failed` observation with a short critique and the loop continues; a pass lets the run end. Verifiers check the outcome (tests, a checker function), not only the code. Emits `verification_passed` and `verification_failed`.
Tests: a failing verifier prevents `finished` and feeds back; a passing one ends the run.

### 3.7 Guardrails and the human-in-the-loop seam (`guards.py`)

```python
class Guard(Protocol):
    def check(self, request: ToolRequest, state: RunState) -> Allow | Deny | Ask: ...
```

Guards form a pipeline of pure checks run in order, stopping at the first `Deny` or `Ask`, before a tool runs (`pre`) and on its output after (`post`, for example a secret in a result or an oversized one). Tool results are untrusted data, wrapped and marked as data in the prompt. `Ask` appends `approval_requested`, the run stops with `awaiting_approval`, and `resume(run_id, decision)` appends `approval_granted` or `approval_denied` and continues from the log. The pause sits between choosing a tool and running it, so nothing executes before approval. A budget guard reads the spend log and denies a model call that would cross a per-run or per-month cap. Emits `guard_decision` and the approval events.
Tests: a deny stops the tool and is logged; an ask pauses the run; kill the process, resume, and the run continues; the cap denies at the boundary with exact decimal arithmetic.

### 3.8 Observability (`observability/`)

```python
def spans(events: Iterable[Event]) -> list[Span]: ...             # trace.py
def price(usage: Usage, table: PricingTable) -> Decimal: ...      # cost.py
```

Spans are derived from the event log and never written separately: a run is a trace, and each turn, model call and tool call is a span with timing and attributes, using the OpenTelemetry `gen_ai.*` attribute names where they exist. Durations use `time.monotonic()`. `cost.py` tallies tokens per call and per run and prices them from `pricing.json`. Emits no events of its own.
Tests: spans from a recorded log are stable; pricing matches hand-computed values; cache read and cache write tokens are priced separately.

## 4. Layout

```
src/loomcore/
  events.py      # envelope, kinds, writer, reader, snapshot, replay
  loop.py        # step, Driver, ports, retry policy, escalation
  tools.py       # registry
  sandbox.py     # sandbox and rule evaluation
  workspace.py   # filesystem workspace
  context.py     # pack, compaction, formatter
  prompt.py      # system-prompt builder
  guards.py      # guard pipeline, budget guard
  verify.py      # verifier hook
  observability/
    trace.py
    cost.py
tests/
evals/           # the scripted three-step task
```

## 5. Tests and the eval

Unit tests sit next to the behaviour they prove. The event log also gets [Hypothesis](https://hypothesis.readthedocs.io/en/latest/stateful.html) stateful tests (random appends, snapshots and replays always agree with the fold) and a truncation test (cut the file at a random byte; replay still gives a consistent earlier state).

The eval is a scripted three-step task driven by a fake model that replays canned responses. It runs in CI like a test and must reach `finished` with a complete, replayable log.

## 6. Done for v0.1

A second repository can `import loomcore`, register a tool and run a multi-step task to completion, with the full event log persisted and replayable. Every component has its `Protocol` and tests, the scripted eval is green in CI, and the cost module prices a recorded log from tokens alone.
