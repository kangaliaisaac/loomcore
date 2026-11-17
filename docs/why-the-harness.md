# The harness is the engineering

An agent is a model plus a harness. The model reasons. The harness decides when a run starts, what the model is shown, what it may touch, when it must stop and what is left behind when it does. `loomcore` is that second half, kept small, and this is what building it taught me about why it deserves the attention the model usually gets.

## What it is

A Python library with eight parts: a system-prompt builder, a tool registry, a sandbox, a filesystem workspace, a context store, a self-verification hook, guardrails with a human-approval seam, and observability with token-based cost. They sit around a loop that is a pure function from a state and an event to the next state and the effects to perform. Everything the loop does is appended to a log, and the log is the only record of a run. A run's state is not kept anywhere else; it is _folded_ from the log.

## The hard part: a run that can end on purpose

A demo agent loops until the model stops talking. A useful one has to stop for reasons the model does not control, and then be able to continue. That is three requirements that all lean on the same thing.

- **Stopping.** Turns, time, output size and two budget caps are part of the core, not options. Reaching one pauses the run with the reason named; you resume it with the limits you choose.
- **Resuming.** A crash between a tool call and its result must not lose work, or repeat a write without asking. That is only possible if the log says exactly what happened, so resume reads the log, repairs the spend record, re-runs read-only calls (a tool says whether it is read-only) and asks a human before re-running any other.
- **Trusting what comes back.** A tool result is data, not an instruction. It is rendered between tags that carry the call's id, and `prompt.DATA_RULE`, which you add to your system prompt, says data ends only at the closing tag with the same id. Without that rule the tags protect nothing, and the library says so rather than adding it silently.

None of this is about the model. A better model does not make a crash recoverable or a budget cap enforceable. Those are properties of the harness, and they decide whether an agent can be left to run.

## The decision I am proudest of

Making the log the single source, and deriving everything else from it. The run's state, the trace of spans, the cost of each call, the window the model is shown on each turn: all are functions of the log. That sounds like bookkeeping. It was a bet that the cost of replaying would be repaid by never having two sources of truth, and it paid off in places I did not plan for. The eval can check that a log replays to the same state, rebuilds every request the model saw, and resumes from a cut. The cost is computed from recorded tokens with a versioned price table; the token counts are the facts and the dollar figure is derived. The context store's behaviour can be scored offline, by replaying a recorded run.

## What building it taught me

**A copy of the driver drifts.** I wrote an eviction bench that scores how well the context store chooses what to keep, against furthest in future (the best possible when the future is known, pages are one size and each fits in the budget; a strong baseline otherwise). The first version was a plausible re-implementation of the driver's windowing, and independent review found it wrong in ways that changed the numbers: it dropped what the driver would not have, it left out clearing, and it numbered pages differently from the driver, which changes how the store ranks them. The fix was not to be more careful. It was a test that runs the driver's real window function turn by turn on a log of the same exchanges and compares what each has dropped and cleared, plus a test written to fail if the numbering is wrong. If a second implementation must exist, make a test hold the two together.

**A convenient import can break a promise.** Re-exporting the driver at the top of the package would have made every component load the driver, which breaks the guarantee that each component works without it. A test that imports one component and checks the driver is not loaded caught it, and the public names load on first use instead.

**Honest numbers need their caveats.** On a seeded workload (an agent that keeps coming back to a few files) the store's rule faults on 201 of 264 repeat requests and furthest in future on 44. The gap is real but it is not all policy quality: the baseline clears nothing and keeps no placeholders, so it has the whole budget, and a hit does not refresh a page's age, so the store clears the pages the agent keeps returning to first. The workload is a stand-in, not a measured run. I would rather publish a number with that sentence than a better-looking one without it.

## What is next

The summariser is specified and not built: when clearing is not enough, a model call should shrink the oldest exchanges instead of dropping them. The token counters are the caller's choice: `loomcore` ships offline estimates, and you can pass exact ones where a cap must hold. A counter that ships with it, exact and backed by the provider's count endpoint, would make the exact count the easy choice; even then a cap can be overshot by one call per concurrent run. And four sibling projects (`filingstrawl`, `finanalyst`, `harnessbench` and `streamsense`) will import `loomcore`. They are the real test of whether the seams are in the right place.
