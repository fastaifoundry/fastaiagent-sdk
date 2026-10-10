# AutoLLM

**AutoLLM** (`fastaiagent.optimize`) is eval-driven prompt optimization. Where
`harden()` *recommends* prompt fixes, **AutoLLM closes the loop**: it proposes a
change, applies it to a fresh agent, re-evaluates, keeps the best, and repeats —
until the score stops improving or a budget runs out. A held-out split guards the
winner against overfitting.

!!! tip "Start from the mental model, or a recipe"
    [How AutoLLM works](autollm-how-it-works.md) explains the loop one diagram at a
    time — the split, what the proposer sees, isolation, what counts as a failure,
    the two judges, the budget — with a real run behind every claim.
    [AutoLLM Recipes](autollm-recipes.md) maps situations — a live agent breaking
    rules only your labels know, a move to a cheaper model, a judge you can't
    trust — to runnable examples. The [AutoLLM Closed Loop](../flagships/autollm-closed-loop.md)
    flagship runs the whole thing end to end: traces → dataset → eval → AutoLLM →
    registry version → CI gate → production.

It tunes the **system prompt** by default, and can also tune **few-shot
examples** and **which learned-memory facts to inject** when you opt in — greedy
coordinate ascent, cycling the active levers one per round. The SDK's answer to
LangSmith's *Promptim* / DSPy's `BootstrapFewShot` + metaprompt optimizers, built
on the `evaluate()` you already use.

This is the OSS on-ramp: standard prompt optimization grounded in your own eval
data, end to end in one SDK. A runnable, real-LLM walkthrough lives in
`examples/autollm/`.

!!! note "Scope"
    AutoLLM tunes the system prompt (default) plus, opt-in, few-shot examples and
    which learned facts to inject — all on the **cold-eval** path. Runs are
    persisted and viewable in the Local UI under **AutoLLM**. Replay-grounded
    scoring (forking a production trace and rerunning from real operational state)
    is the Enterprise complete-loop capability — the `score_candidate` seam is its
    drop-in point.

!!! note "Optimizing on your traces = agent-attributable cases only"
    When you build the eval set from traces
    ([curation](curation.md)), AutoLLM optimizes only on **agent-quality** failures,
    not infrastructure failures: a run that infra-errored (endpoint 500, timeout)
    and produced no usable output is dropped, not curated as a gold target, and a
    case that errors *during* scoring is never shown to the prompt proposer as a
    failure to fix. So the optimizer never chases a fault the agent can't fix — but
    the errored case still counts against the candidate's score (see
    [When a case errors](#when-a-case-errors)). Runnable walkthrough:
    `examples/80_curate_from_traces.py`.

## Quickstart

```python
import fastaiagent as fa

agent = fa.Agent(name="capitals", system_prompt="You answer questions.", llm=fa.LLMClient())

report = fa.optimize(
    agent,
    "cases.jsonl",                 # Dataset | path | list[dict] with input/expected_output
    scorers=["exact_match"],
    config=fa.OptimizeConfig(max_iterations=5, patience=2),
)

print(report.summary())
better_agent = report.apply_to(agent)   # a fresh agent with the winning prompt
```

`optimize()` is the sync wrapper; `aoptimize()` is the async implementation (it's
a minutes-to-hours operation — prefer async in apps).

## How the loop works

```
split (seeded) → train / dev / holdout
baseline scored on dev            (already at target_score → stop here)
repeat (cycling active levers: instructions → fewshot → memory):
  propose N candidate variants of the active lever, on top of the current best
  score each on dev
  keep the best if it beats the current best by ≥ min_delta   (else → patience)
  stop on: patience | max_iterations | target_score | budget | proposer_failed
holdout guard: re-score the winner on the held-out split; revert to baseline
               if it regressed beyond holdout_regression_tol
```

"Optimized" means *hill-climbed until no improvement or budget exhausted* — the
same operational definition Promptim and DSPy use. The **holdout guard**, not the
search, is what makes the result trustworthy rather than overfit: the holdout
split never influences selection, so the reported lift is on data no candidate
was tuned against. By construction the winner is **never worse than baseline**.

!!! info "Algorithm"
    AutoLLM uses **greedy coordinate ascent** (Promptim-style keep/revert, one
    lever per round) with a **metaprompt / reflective proposer** — the optimizer
    reads the dev failures and writes a revised prompt. This is the same algorithm
    family as **LangSmith Promptim** and **DSPy** (`BootstrapFewShot` + metaprompt
    optimization). The joint-Bayesian-search variant — DSPy's **MIPRO** — searches
    instructions and demos together; it's a documented upgrade path
    (`strategy="mipro"`) rather than the default, since coordinate ascent gives a
    single-lever cause for every accepted step and avoids MIPRO's cost multiplier.

    Each failing case shown to the proposer includes its **expected output** and the
    **scorer's reason** (e.g. `"got 1120, expected 1120000"`), not just the input and
    the wrong answer. This is what lets AutoLLM optimize **extraction and
    structured-output** tasks, where the fix is an output *convention* (scale, sign,
    formatting) the proposer can only infer by seeing what correct looks like — for
    example recovering *"values are in thousands → multiply by 1,000; parentheses are
    negative; answer with the number alone"* when pulling figures from financial
    tables. (Added in 1.38.0; classification never needed it because the label space
    is small enough to guess.)

## The levers

- **`instructions`** — rewrites the system prompt. The proposer reuses the
  failure analysis behind `harden()` but lives in `fastaiagent.optimize`;
  `harden()` and the rest of the `eval` API are unchanged. The proposer is shown
  up to 40 failing train cases per round. `optimize()` raises `ValueError` for an
  agent with a callable (dynamic) `system_prompt` while this lever is active —
  leave `"instructions"` out of `levers` for such an agent.
- **`fewshot`** — bootstraps few-shot examples (DSPy `BootstrapFewShot`): gold
  `(input, expected_output)` pairs from the **train** split (plus
  `curate_from_traces(filter="favorites")`), filling any gap by running the agent
  and metric-filtering its passing outputs. Demos are injected via a `FewShotBlock`.
  **No demo ever carries a dev or holdout input** — a favorite trace whose input is
  a scored case is skipped, so an eval set curated from your favorites can't hand
  the agent the answers it is scored on.
- **`memory`** — tunes *which subset* of the agent's learned facts to inject,
  via a confidence/recency ablation. It reads the facts **where the agent's
  memory reads them**: a `Memory`'s `location` and `project_id` (its global
  tier), or a `PersistentFactBlock`'s own store, scope and `project_id`; an
  agent with neither uses `local.db` at `("agent", <agent name>)`. Pure
  **selection** — it never creates, edits, or deletes facts, so the audit chain
  is untouched. Injected through a `PersistentFactBlock` backed by an allowlist
  over that same store. With no facts there it's **skipped** (recorded distinctly
  from a reject). `fastaiagent learn` writes to `local.db`, so for an agent whose
  memory lives in Postgres or Redis, put the facts in that store.

The default is **prompt-only** (`levers=("instructions",)`) — the cheapest entry
point (few-shot adds a bootstrap pass; memory needs `fastaiagent learn` to have
run). Opt into more with e.g. `levers=("instructions", "fewshot", "memory")`.

**A stateless agent stays stateless.** The few-shot and memory levers carry their
examples and facts in a memory block. An agent built with no `memory` gets those
blocks and nothing else — it keeps no conversation, during the run (each eval case
is scored on its own) and after `report.apply_to(agent)` (one user's request never
reaches the next user's prompt). Before 1.87.0 the wrapper kept a conversation
window, so a tuned stateless agent remembered every earlier run.

### Memory-bearing agents

Each candidate evaluation gets an **isolated copy** of the agent's memory
(`block.isolated_copy()`: shares external handles like the `llm`, resets
in-process state) so one candidate's turns never bleed into another's. Agents
with `StaticBlock` / `PersistentFactBlock` / `PlaneFactBlock` / `SummaryBlock` /
`FactExtractionBlock(persist=False)` are supported. **`VectorBlock` and
`FactExtractionBlock(persist=True)` are excluded** — they write to an external
store during a run, so sharing them bleeds candidates; `optimize()` refuses
unless you pass `allow_writable_memory=True` (accepting the bleed).

#### Agents built on `Memory`

An agent whose memory is a [`Memory`](../agents/memory.md) optimizes the same way, and the agent you get back still has a `Memory` — with the same `window`, `location`, `agent_id` and per-user routing:

```python
from dataclasses import dataclass

import fastaiagent as fa

agent = fa.Agent(
    name="support",
    system_prompt="You answer billing questions.",
    llm=fa.LLMClient(),
    memory=fa.Memory(agent_id="support", user_id=lambda ctx: ctx.state.user_id, window=20),
)

report = fa.optimize(
    agent, "cases.jsonl", scorers=["exact_match"],
    config=fa.OptimizeConfig(levers=("instructions", "fewshot", "memory")),
)
better = report.apply_to(agent)   # better.memory is a Memory, one window per user


@dataclass
class Session:
    user_id: str


better.run("Why was I charged twice?", context=fa.RunContext(state=Session(user_id="alice")))
```

A runnable version, with the memory lever over facts in the agent's own store: [`examples/101_optimize_memory_agent.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/101_optimize_memory_agent.py).

What carries over and what is fresh:

- **Configuration carries over:** the store, `window`, `agent_id`, `max_users`, and the per-user resolver — every user still gets their own window.
- **Windows start empty:** each candidate, and the returned agent, starts with no conversations in memory. Durable facts are in the store and are read as usual.
- **The levers reach every user:** the few-shot demos and the selected facts are injected for each user, and for callers with no user.
- **The memory lever works on global facts:** it selects among the agent's global facts (`agent_id=`, or the agent's name when unset) in the `Memory`'s own store and `project_id` — never among one user's facts.

Memory that **writes during a run** can't be isolated per candidate, so, as with the blocks above, `optimize()` refuses it unless you pass `allow_writable_memory=True`:

| Keyword | During optimize |
|---|---|
| `learn=` | refused — it writes learned facts to the store on every turn |
| `recall=<VectorStore>` | refused — every candidate would write to the same store |
| `recall="auto"` | allowed — each candidate builds its own in-process index |
| `summarize=`, `semantic=` | allowed |

## Configuration

```python
fa.OptimizeConfig(
    max_iterations=8,            # hard cap on rounds
    patience=3,                  # stop after N non-improving rounds
    target_score=None,          # stop early once dev reaches this
    candidates_per_iteration=3, # proposals per round
    min_delta=0.01,             # improvement smaller than this = "no improvement"
    splits=(0.5, 0.25, 0.25),   # train / dev / holdout
    holdout_regression_tol=0.0, # revert if holdout drops more than this
    seed=0,                     # deterministic split
    primary_metric=None,        # scorer name to select on (default: overall pass-rate)
    max_eval_runs=None,         # hard cap on evaluation passes (see Cost)
    max_judge_calls=None,       # hard cap on model-backed scorer calls (see Cost)
    selection_judge=None,       # an LLM judge used *inside* the loop
    audit_judge=None,           # an LLM judge used *only* on the holdout guard
    levers=("instructions",),   # default: prompt only — add "fewshot" and/or "memory"
    allow_writable_memory=False,  # opt in to memory that writes during a run (bleed risk)
)
```

### The two-judge guard

For agents graded by a deterministic scorer, set `primary_metric` and you're
done. For **reference-free agents** (research, summarization, KYC narratives)
selection *is* an LLM judge — and optimizing against the same judge you report is
reward-hacking waiting to happen. Pass distinct judges:

```python
from fastaiagent.eval import GEval

cfg = fa.OptimizeConfig(
    selection_judge=GEval(criteria="answer quality"),         # drives accept/reject
    audit_judge=GEval(criteria="answer quality", name="audit",  # different prompt/model
                      evaluation_steps=[...]),
)
```

If you leave `audit_judge` unset, it falls back to the selection judge **with a
warning** — fine for a first pass, not for a number you'll quote. Judges are
ordinary `Scorer`s; they're composed into the scorers list (and deduped, so a
judge you already pass in `scorers` isn't billed twice).

Because judges are deduped by **name**, an `audit_judge` must not share its name
with a different scorer in `scorers` — results are keyed by name, so the holdout
would be scored by that other scorer instead. `optimize()` raises `ValueError` up
front when they clash: `LLMJudge` defaults to `"llm_judge"` and `GEval` to
`"g_eval"`, so give the audit judge its own `name=`, as above. Passing the *same*
judge object in `scorers` and as `audit_judge` makes it drive selection as well, and
`optimize()` warns.

## Reading the report

`OptimizationReport` mirrors `HardeningReport` (`.summary()`, `.to_dict()`) and
adds the score trajectory and an applyable winner:

```
Optimization — capitals (stopped: patience)
============================================================
baseline   dev=0.600
 iter 1 [instructions]  dev=0.800 (+0.200)  ACCEPT  — answer with only the place name
 iter 2 [instructions]  dev=0.800 (+0.200)  reject
------------------------------------------------------------
best        dev=0.800
holdout     best=0.750 (baseline=0.600, Δ+0.150) → winner kept
```

- `report.best_candidate.system_prompt` — the winning prompt.
- `report.apply_to(agent)` — a copy of your agent with the winning levers applied:
  same class, tools, guardrails, middleware and agent path. The original is never
  mutated. A changed prompt drops `prompt_slug`, since the registry prompt it names
  is no longer what the agent runs.
- `report.trajectory` — every candidate scored, with lever attribution and the
  number of dev cases that `errored`.
- `report.improved` — did the winner beat baseline and survive the holdout guard?
- `report.stopped_reason` — `patience`, `max_iterations`, `target_score`, `budget`,
  `proposer_failed` or `no_active_levers`, with `+reverted` appended when the
  holdout guard reverted the winner.
- `report.proposer_errors` — each time the prompt proposer could not run or its
  reply could not be read.

### When a case errors

A case that raises instead of answering — a guardrail block, `MaxIterationsError`,
a provider error — **counts as a failure** in the candidate's dev and holdout
scores (score 0 on every metric). `evaluate()` leaves such a case out of its own
pass rate, but selecting on that would let a candidate that crashes on its hard
cases outscore one that answers them. The summary shows the count:

```
 iter 1 [instructions]  dev=0.500 (+0.500)  reject  [2 errored]
```

An errored case is still never shown to the prompt proposer as a failure to fix.
A flaky provider therefore costs a candidate points rather than handing it a win:
use a model client with retries for long runs.

### When the proposer fails

If the prompt proposer can't run — an unknown model, an auth error, a reply that
isn't the requested JSON — the round is recorded as a skipped step, the error is
logged as a warning and kept in `report.proposer_errors`, and a run that ends
because of it stops with `proposer_failed`, not `patience`:

```
Optimization — capitals (stopped: proposer_failed)
============================================================
baseline   dev=0.000
 iter 1 [instructions] SKIPPED — proposer failed: LLMProviderError: OpenAI API error 404 …
 iter 2 [instructions] SKIPPED — proposer failed: LLMProviderError: OpenAI API error 404 …
------------------------------------------------------------
best        dev=0.000
holdout     best=0.000 (baseline=0.000, Δ+0.000) → winner kept
proposer failed 2x — LLMProviderError: OpenAI API error 404 …
```

## Persistence & the UI

When `optimize(..., persist=True)` (the default), the run is recorded to the
local `local.db` and surfaces in `fastaiagent ui` under **AutoLLM** — no
extra wiring. Two tables hold the record:

- **`optimize_runs`** — one parent row per run: baseline/best dev scores, the
  holdout-guard scores, `stopped_reason`, `reverted`, the `seed`, the active
  `levers`, the winning `Candidate` as JSON (for reproducibility), and in
  `metadata` the agent's original system prompt (`baseline_system_prompt`) and any
  `proposer_errors`.
- **`optimize_iterations`** — one row per trajectory point: `iteration`, `lever`,
  `dev_score`, `accepted`/`skipped`, `rationale`, and an `eval_run_id`.

The `eval_run_id` is the key to the **drill-down**. Every candidate is scored by
a real `aevaluate(persist=…)` call, so it already lands in `eval_runs` /
`eval_cases` with traced runs. The iteration row just *links* to that existing
eval run — optimize stores **no duplicate eval data**. In the UI you can follow:

```
AutoLLM → a run → trajectory row → its eval run → the per-case traces
```

The view is read-only and refresh-based (REST, no live streaming). Open a run to see:

- **Summary** — baseline and best dev scores, the holdout score, and why the run
  stopped.
- **Winner** — the winning system prompt next to the prompt the run started from
  (each with a copy button), the few-shot examples and learned facts it selected,
  and any proposer failures. A reverted run, or one where nothing beat the
  baseline, says the agent keeps its original configuration.
- **Trajectory** — `baseline → accepted/skipped steps → holdout-guarded winner`
  with per-iteration lever attribution; click any row through to the eval that
  produced its score.

Each run targets a single agent, so with several agents the list shows one row
per run tagged by `agent_name`; an **agent filter** appears once more than one
agent has runs (backed by `GET /api/optimizes?agent=…`).

Persistence is gated by the same `persist` flag that gates per-candidate evals,
so `optimize(..., persist=False)` writes nothing to `optimize_runs` /
`optimize_iterations` (and skips the per-candidate `eval_runs` writes too).

## CLI

```sh
fastaiagent optimize \
  --agent myapp.py:agent \
  --dataset cases.jsonl \
  --scorers exact_match \
  --max-iterations 5 \
  --levers instructions,fewshot \
  --judge "is the answer correct and concise" \
  --audit-judge "is the answer correct, complete and concise" \
  --out winning_prompt.txt
```

- `--agent` is a `path/to/file.py:attr` or `pkg.module:attr` that resolves to an `Agent`.
- `--levers` is a comma-separated subset of `instructions`, `fewshot` and `memory`
  (default `instructions`).
- `--judge` adds an LLM judge (a criteria string) as the selection scorer;
  `--audit-judge` adds a distinct one used only on the holdout guard.
- `--out` writes the winning system prompt before the summary is printed.

## When not to use it

- **Tiny datasets (< ~15 cases)** can't form a meaningful 3-way split — run
  `harden()` once instead. `optimize()` warns below 15 and errors below 3.
- **`VectorBlock`-bearing agents** can't be isolated per candidate (the block
  writes to an external store mid-run) — `optimize()` refuses unless you pass
  `allow_writable_memory=True`. Other memory blocks are isolated automatically.
- **Tool/retrieval-bound agents** — if quality is dominated by tool correctness
  rather than the prompt, fix the tools first.
- **A `Supervisor`, `Swarm` or `Chain`** — `optimize()` takes an `Agent` and raises
  `TypeError` for anything else. Optimize the agent behind each step.

## Cost

The bill compounds: `iterations × candidates × dev-size × judge-calls`. Two hard
caps bound it:

- **`max_eval_runs`** counts every evaluation pass: the baseline, each train
  re-score, each candidate, the few-shot teacher pass, and the holdout guard's
  passes.
- **`max_judge_calls`** counts one call per case for every model-backed scorer in
  a pass — `LLMJudge`/`GEval`, `DecisionJudge`, and the built-in RAG, agent,
  session and safety metrics — whether passed in `scorers` or as
  `selection_judge`/`audit_judge`. A custom `Scorer` that calls a model itself
  isn't counted.

The loop holds back what the holdout guard needs, so the guard always runs and
neither total ever passes its cap. Caps too small for the baseline plus the guard
raise `ValueError` before anything runs. Select on a cheap deterministic scorer,
reserve the LLM judge for the holdout audit, and let `patience` / `min_delta`
stop early on noise.
