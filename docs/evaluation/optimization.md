# Prompt & Few-shot Optimization

`harden()` *recommends* prompt fixes. **`optimize()` closes the loop**: it
proposes a change, applies it to a fresh agent, re-evaluates, keeps the best, and
repeats — until the score stops improving or a budget runs out. A held-out split
guards the winner against overfitting.

It tunes the **system prompt** by default, and can also tune **few-shot
examples** when you opt in — greedy coordinate ascent, cycling the active levers
one per round. The SDK's answer to LangSmith's *Promptim* / DSPy's
`BootstrapFewShot` + metaprompt optimizers, built on the `evaluate()` you already
use.

!!! note "Scope"
    Tunes the system prompt (default) plus, opt-in, few-shot examples and which
    learned facts to inject — all on the cold-eval path. Persisted runs in the
    Local UI build on the same data model (a later phase) without reshaping it.

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
baseline scored on dev
repeat (cycling active levers: instructions → fewshot → memory):
  propose N candidate variants of the active lever, on top of the current best
  score each on dev
  keep the best if it beats the current best by ≥ min_delta   (else → patience)
  stop on: patience | max_iterations | target_score | budget
holdout guard: re-score the winner on the held-out split; revert to baseline
               if it regressed beyond holdout_regression_tol
```

"Optimized" means *hill-climbed until no improvement or budget exhausted* — the
same operational definition Promptim and DSPy use. The **holdout guard**, not the
search, is what makes the result trustworthy rather than overfit: the holdout
split never influences selection, so the reported lift is on data no candidate
was tuned against. By construction the winner is **never worse than baseline**.

## The levers

- **`instructions`** — rewrites the system prompt. The proposer reuses the
  failure analysis behind `harden()` but lives in `fastaiagent.optimize`;
  `harden()` and the rest of the `eval` API are unchanged. (Skipped for agents
  with a callable/dynamic `system_prompt`.)
- **`fewshot`** — bootstraps few-shot examples (DSPy `BootstrapFewShot`): gold
  `(input, expected_output)` pairs from the **train** split (plus
  `curate_from_traces(filter="favorites")`), filling any gap by running the agent
  and metric-filtering its passing outputs. Demos are injected via a `FewShotBlock`
  and never drawn from dev/holdout (no leakage).
- **`memory`** — tunes *which subset* of the agent's learned facts
  (`MemoryStore.list_active`, populated by `fastaiagent learn`) to inject, via a
  confidence/recency ablation. Pure **selection** — it never creates, edits, or
  deletes facts, so the audit chain is untouched. Injected through a
  `PersistentFactBlock` backed by an allowlist store. Needs facts at the agent's
  scope; with none it's **skipped** (recorded distinctly from a reject).

The default is **prompt-only** (`levers=("instructions",)`) — the cheapest entry
point (few-shot adds a bootstrap pass; memory needs `fastaiagent learn` to have
run). Opt into more with e.g. `levers=("instructions", "fewshot", "memory")`.

### Memory-bearing agents

Each candidate evaluation gets an **isolated copy** of the agent's memory
(`block.isolated_copy()`: shares external handles like the `llm`, resets
in-process state) so one candidate's turns never bleed into another's. Agents
with `StaticBlock` / `PersistentFactBlock` / `PlaneFactBlock` / `SummaryBlock` /
`FactExtractionBlock` are supported. **`VectorBlock` is excluded** — it writes to
an external store during a run, so sharing it bleeds candidates; `optimize()`
refuses unless you pass `allow_writable_memory=True` (accepting the bleed).

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
    max_eval_runs=None,         # cost governor: cap candidate evaluations
    max_judge_calls=None,       # cost governor: cap judge invocations
    selection_judge=None,       # an LLM judge used *inside* the loop
    audit_judge=None,           # an LLM judge used *only* on the holdout guard
    levers=("instructions",),   # default: prompt only — add "fewshot" and/or "memory"
    allow_writable_memory=False,  # opt in to VectorBlock agents (bleed risk)
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
- `report.apply_to(agent)` — a fresh agent with it (the original is never mutated).
- `report.trajectory` — every candidate scored, with lever attribution.
- `report.improved` — did the winner beat baseline and survive the holdout guard?

## CLI

```sh
fastaiagent optimize \
  --agent myapp.py:agent \        # module:attr resolving to an Agent
  --dataset cases.jsonl \
  --scorers exact_match \
  --max-iterations 5 \
  --judge "is the answer correct and concise" \   # optional LLM selection judge
  --out winning_prompt.txt
```

## When not to use it

- **Tiny datasets (< ~15 cases)** can't form a meaningful 3-way split — run
  `harden()` once instead. `optimize()` warns below 15 and errors below 3.
- **`VectorBlock`-bearing agents** can't be isolated per candidate (the block
  writes to an external store mid-run) — `optimize()` refuses unless you pass
  `allow_writable_memory=True`. Other memory blocks are isolated automatically.
- **Tool/retrieval-bound agents** — if quality is dominated by tool correctness
  rather than the prompt, fix the tools first.

## Cost

The bill compounds: `iterations × candidates × dev-size × judge-calls`. Use
`max_eval_runs` / `max_judge_calls` as hard governors, select on a cheap
deterministic scorer and reserve the LLM judge for the holdout audit, and let
`patience` / `min_delta` stop early on noise.
