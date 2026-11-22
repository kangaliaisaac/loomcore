"""How well does the context store choose what to keep? (CLRS 15.4, offline caching)

The model asks for the same thing again when a window no longer shows what it needed.
Read a run's tool calls as a string of requests, one per call, keyed by what was
asked and what came back. Replay it twice with one token budget:

- the context store's rule as the driver applies it (`window.build_window`): the task
  and the newest page are pinned; `pack` clears the oldest results when the window is
  over budget and drops the rest by value per token; `compact` clears more once the
  window reaches `compact_at` of the budget, down to `compact_to`; what earlier
  windows dropped or cleared stays so; a window whose pinned items alone are over the
  budget is `over`, sent as it was and recording nothing;
- furthest in future (CLRS 15.4): with the whole string known, evict the page needed
  furthest ahead. It is the best possible when pages are the same size and every page
  fits in the budget, and only a strong baseline otherwise (the best possible for
  mixed sizes is NP-hard).

A *fault* is a request for something asked before whose result is no longer shown in
full: dropped, or cleared. The first request for anything is a miss for any policy and
is not counted. Fewer faults is better; the gap is what the store loses to not
knowing the future. Furthest in future keeps no placeholders and clears nothing, so it
has the whole budget; the store spends some of it on the stubs of cleared results.

Pages are sized as the driver counts them: the whole exchange, the model's tool call
and the result block `conversation.tool_result` renders, with `estimate_tokens`; a
cleared page is the same exchange as `clear_results` leaves it. The model's own text
and thinking around the call are not counted, so on a real log a page is smaller than
the driver's item and the budget is approximate.
The task, read from the log's `run_started`, takes its tokens of the budget, pinned,
as in the driver, and compaction uses the run's own `compact_at` and `compact_to`
(a run that changed them on a resume is refused). Not modelled: the plan exchange and
any hint (calls to `update_plan` and `finish` are left out of the string), the
exchange of a refused call, and a model turn with no tool call. The last two take
budget and push later item numbers up in the driver, so on a log that has them the
bench's numbering and budget are a little off.

A request for something the window still shows is a hit in the replay and makes no
page. In a real log the driver built an exchange for it all the same, so after the
first such request the replay's windows drift from the driver's. The report counts
them ("the window still showed"): a real log's numbers are only close when that is
zero or small. The seeded workload is a model of requests, and treats them as hits
by design.

    uv run --no-sync python bench/eviction.py                    # a seeded workload
    uv run --no-sync python bench/eviction.py BUDGET LOG [LOG..] # real runs' logs

BUDGET is the history budget, as the driver passes it to `pack`: `context_budget` less
the system prompt and the tool schemas, which the bench does not see, in the units of
`estimate_tokens` (a run that counted with another counter has its own units).

A log's string is one request per tool call that succeeded (a refused call and
`finish` are left out), keyed by the call and its result, so a re-read after a write,
whose result is new, is a first request for every policy. An exchange holding several
calls counts as one page per call. A log that holds more than one run (a second
`run`, or a fork) is refused: its calls are not one run's string. It is
the string the driver's own window produced, so a policy that cleared more would have
been asked for more: the numbers compare policies on one recorded string, they do not
predict another run.
"""

import hashlib
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from loomcore import estimate_tokens, finish, read_events
from loomcore.context import PLAN_TOOL, Cleared, Item, clear_results, compact, pack
from loomcore.conversation import tool_result
from loomcore.events import (
    Event,
    JsonObject,
    RunResumed,
    RunStarted,
    ToolCompleted,
    ToolRequested,
    make_event,
)
from loomcore.limits import Limits

_DEFAULTS = Limits.model_fields  # the fractions at which compaction starts and stops
COMPACT_AT: float = _DEFAULTS["compact_at"].default
COMPACT_TO: float = _DEFAULTS["compact_to"].default
SEEDED_BUDGET = 20_000  # tokens: the budget the seeded workload is scored at


# Calls the driver treats specially: `finish` ends the run and the plan's exchange is
# pinned, so neither is an ordinary page.
_NOT_PAGES = {finish.NAME, PLAN_TOOL}


@dataclass(frozen=True)
class Ref:
    """One request: what was asked and answered (`key`), the tokens its exchange
    takes, and the tokens it takes once its result is cleared."""

    key: str
    tokens: int
    stub: int


@dataclass(frozen=True)
class Replay:
    """The store's rule on a string: its faults, and the pages (by position in the
    string) that were dropped or cleared by the end."""

    faults: int
    dropped: frozenset[int]
    cleared: frozenset[int]
    created: tuple[int, ...]  # the requests that reached the world and made a page
    # What had been dropped and cleared after each of those pages' windows, so a test
    # can compare the state every decision was made on, not only the end.
    steps: tuple[tuple[frozenset[int], frozenset[int]], ...]


@dataclass(frozen=True)
class Score:
    requests: int
    compulsory: int  # distinct keys: the first request for each, a miss for anyone
    policy: int  # faults of the context store's rule
    furthest: int  # faults of furthest in future
    served: int  # requests the window still showed in full: hits, no page made

    @property
    def avoidable(self) -> int:
        """Requests that were not a first ask: the ones a policy can get right."""
        return self.requests - self.compulsory


def completed(tool_use_id: str, text: str) -> ToolCompleted:
    """A successful tool result, as the driver logs one from a tool."""
    event = make_event(
        "tool_completed",
        seq=0,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        session_id="bench",
        run_id="bench",
        tool_use_id=tool_use_id,
        ok=True,
        content=text,
        duration_ms=0,
        cached=False,
        from_tool=True,
    )
    assert isinstance(event, ToolCompleted)
    return event


def exchange(name: str, args: JsonObject, done: ToolCompleted) -> list[JsonObject]:
    """The two messages one tool call leaves in the history, as the driver renders."""
    use: JsonObject = {
        "type": "tool_use",
        "id": done.tool_use_id,
        "name": name,
        "input": args,
    }
    return [
        {"role": "assistant", "content": [use]},
        {"role": "user", "content": [tool_result(done)]},
    ]


def ref_for(key: str, name: str, args: JsonObject, done: ToolCompleted) -> Ref:
    messages = exchange(name, args, done)
    return Ref(
        key,
        estimate_tokens(messages),
        estimate_tokens(clear_results(messages)),
    )


def references(events: Sequence[Event]) -> list[Ref]:
    """The successful tool calls of a log, in order, as requests. A log that holds
    more than one run (a second `run`, or a fork): `ValueError`."""
    if len({e.run_id for e in events}) > 1:
        raise ValueError("the log holds more than one run; score the log of one run")
    results = {
        e.tool_use_id: e for e in events if isinstance(e, ToolCompleted) and e.ok
    }
    refs = []
    for event in events:
        if not isinstance(event, ToolRequested) or event.name in _NOT_PAGES:
            continue
        done = results.get(event.tool_use_id)
        if done is None:
            continue
        text = (
            done.content if isinstance(done.content, str) else json.dumps(done.content)
        )
        key = f"{event.call_hash}:{hashlib.sha256(text.encode()).hexdigest()[:12]}"
        refs.append(ref_for(key, event.name, event.input, done))
    return refs


def log_task_tokens(events: Sequence[Event]) -> int:
    """The tokens the log's task takes in the window, as the driver counts it."""
    for event in events:
        if isinstance(event, RunStarted):
            return estimate_tokens([{"role": "user", "content": event.task}])
    return 0


def replay_policy(
    refs: Sequence[Ref],
    budget: int,
    task_tokens: int = 0,
    compact_at: float = COMPACT_AT,
    compact_to: float = COMPACT_TO,
) -> Replay:
    """The context store's rule on the string, as `window.build_window` applies it:
    each missed request adds a page, the newest page is pinned, and what a window
    drops or clears stays dropped or cleared."""
    task = Item("task", "task", 0, task_tokens, (), pinned=True)
    newest: dict[str, int] = {}  # key -> position of its newest page
    dropped: set[int] = set()
    cleared: set[int] = set()
    pages: dict[int, Item] = {}
    created: list[int] = []
    steps: list[tuple[frozenset[int], frozenset[int]]] = []
    faults = 0
    for order, ref in enumerate(refs):
        last = newest.get(ref.key)
        if last is not None and last not in dropped | cleared:
            continue  # still shown in full: no request to the world
        faults += last is not None
        # The driver numbers the task 0 and each exchange after it 1, 2, 3, with no gap
        # for a request that did not reach the world: `pack` ranks by that number.
        page = Item(
            str(order),
            "exchange",
            len(created) + 1,
            ref.tokens,
            (),
            cleared=Cleared((), ref.stub),
        )
        newest[ref.key] = order
        shown = [_shown(p, cleared) for p in pages.values() if int(p.id) not in dropped]
        packed = pack(shown, budget, [task, replace(page, pinned=True)])
        if not packed.over:  # an over window is sent as it was and records nothing
            packed = compact(packed, compact_at, compact_to)
            dropped |= {int(p.id) for p in packed.dropped}
            cleared |= {int(i) for i in packed.cleared}
        pages[order] = page
        created.append(order)
        steps.append((frozenset(dropped), frozenset(cleared)))
    return Replay(
        faults, frozenset(dropped), frozenset(cleared), tuple(created), tuple(steps)
    )


def _shown(page: Item, cleared: set[int]) -> Item:
    """A page as an earlier window left it: its result cleared, if it was."""
    if int(page.id) not in cleared or page.cleared is None:
        return page
    return replace(page, tokens=page.cleared.tokens, cleared=None)


def policy_faults(
    refs: Sequence[Ref],
    budget: int,
    task_tokens: int = 0,
    compact_at: float = COMPACT_AT,
    compact_to: float = COMPACT_TO,
) -> int:
    return replay_policy(refs, budget, task_tokens, compact_at, compact_to).faults


def optimal_faults(refs: Sequence[Ref], budget: int, task_tokens: int = 0) -> int:
    """Faults of furthest in future: on a miss, evict the page needed furthest ahead
    (never asked again counts as furthest) until the new page fits in what the task
    leaves."""
    room = budget - task_tokens
    next_use = _next_uses(refs)
    cache: dict[str, int] = {}  # key -> index of its next use
    sizes: dict[str, int] = {}
    seen: set[str] = set()
    faults = 0
    for i, ref in enumerate(refs):
        if ref.key in cache:
            cache[ref.key] = next_use[i]
            continue
        faults += ref.key in seen
        seen.add(ref.key)
        while cache and sum(sizes[k] for k in cache) + ref.tokens > room:
            del cache[max(cache, key=lambda k: cache[k])]
        cache[ref.key] = next_use[i]
        sizes[ref.key] = ref.tokens
    return faults


def _next_uses(refs: Sequence[Ref]) -> list[int]:
    """For each request, the index of the next request for the same key."""
    never = len(refs)
    upcoming: dict[str, int] = {}
    out = [never] * len(refs)
    for i in range(len(refs) - 1, -1, -1):
        out[i] = upcoming.get(refs[i].key, never)
        upcoming[refs[i].key] = i
    return out


def score(
    refs: Sequence[Ref],
    budget: int,
    task_tokens: int = 0,
    compact_at: float = COMPACT_AT,
    compact_to: float = COMPACT_TO,
) -> Score:
    replay = replay_policy(refs, budget, task_tokens, compact_at, compact_to)
    return Score(
        len(refs),
        len({r.key for r in refs}),
        replay.faults,
        optimal_faults(refs, budget, task_tokens),
        len(refs) - len(replay.created),
    )


def compaction_fractions(events: Sequence[Event]) -> tuple[float, float]:
    """The run's `compact_at` and `compact_to`. The bench replays one pair for the
    whole string, so a run whose limits changed them on a resume or a fork is refused:
    `ValueError`."""
    seen = set()
    for event in events:
        if isinstance(event, RunStarted | RunResumed) and event.limits is not None:
            limits = Limits.model_validate(event.limits)
            seen.add((limits.compact_at, limits.compact_to))
    if len(seen) > 1:
        raise ValueError("the run's compaction fractions changed on a resume")
    return seen.pop() if seen else (COMPACT_AT, COMPACT_TO)


def score_log(events: Sequence[Event], budget: int) -> Score:
    """A log's score at a history budget, with the run's own task and compaction
    fractions."""
    compact_at, compact_to = compaction_fractions(events)
    return score(
        references(events), budget, log_task_tokens(events), compact_at, compact_to
    )


def seeded_workload(requests: int = 300, seed: int = 0) -> list[Ref]:
    """An agent working on a few files it comes back to: 40 files whose results take
    200 to 4000 tokens, a hot set of six asked for 70% of the time. Seeded, so the
    same every run; it is a stand-in for a real run, not a measurement of one."""
    rng = random.Random(seed)
    pages = [
        ref_for(
            f"file-{n}",
            "read_file",
            {"path": f"src/module_{n}.py"},
            completed(f"t{n}", "x" * (3 * rng.randrange(200, 4001))),
        )
        for n in range(40)
    ]
    hot = pages[:6]
    return [
        rng.choice(hot) if rng.random() < 0.7 else rng.choice(pages)
        for _ in range(requests)
    ]


def report(result: Score, budget: int, source: str) -> str:
    of = f"of {result.avoidable}"
    return (
        f"{source}: {result.requests} requests, budget {budget} tokens\n"
        f"  first asks (a miss for any policy): {result.compulsory}\n"
        f"  faults, the context store's rule:    {result.policy} {of}\n"
        f"  faults, furthest in future:          {result.furthest} {of}\n"
        f"  requests the window still showed:    {result.served}"
    )


def main(argv: Sequence[str]) -> None:
    if not argv:
        budget = SEEDED_BUDGET
        print(report(score(seeded_workload(), budget), budget, "seeded workload"))
        return
    budget = int(argv[0])
    for path in argv[1:]:
        read = read_events(Path(path))
        if read.torn_tail is not None:
            print(f"{path}: an unfinished last line was dropped", file=sys.stderr)
        print(report(score_log(read.events, budget), budget, path))


if __name__ == "__main__":
    main(sys.argv[1:])
