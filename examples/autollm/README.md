# AutoLLM — eval-driven prompt optimization (end to end, real LLM)

**AutoLLM** (`fastaiagent.optimize`) closes the loop `harden()` opens: instead of
only *recommending* fixes, it proposes a change, applies it to a fresh agent,
re-evaluates on a held-out split, keeps the best, and holdout-guards the winner so
it can't overfit. This is the OSS on-ramp: **standard prompt optimization grounded
in your own eval data, end to end in one SDK.**

These examples are fully runnable with a real OpenAI model — **no mocks**:

- **`agent.py`** — the minimal on-ramp: a weak sentiment prompt fails strict
  `exact_match`; AutoLLM recovers a one-word-output format fix (`gpt-4o-mini`).
- **`financials.py`** — a real extraction task: `gpt-4o` pulls values from financial
  tables but scores **0%** because it never applies the `"(in thousands)/(in millions)"`
  scale; AutoLLM recovers the convention (**0% → 86% dev, 100% holdout**), graded by a
  custom `NumericMatch` scorer. Shows AutoLLM works on more than classification, and
  that even a strong model needs a convention your data encodes. *(Requires ≥ 1.38.0.)*

## What `agent.py` does

1. Builds a sentiment classifier with a deliberately **weak** prompt
   (`"You classify the sentiment of a customer review."`) — it sets the task but
   no output *format*, so the model answers in full sentences
   (`"The sentiment is positive."`) and **fails strict `exact_match`** against the
   one-word gold label.
2. Runs `optimize(agent, dataset, ["exact_match"], persist=True)` — AutoLLM reads
   the dev failures, proposes prompt rewrites, scores each candidate, and keeps the
   best (the instructions lever, greedy coordinate ascent).
3. Holdout-guards the winner, prints the trajectory, applies the winning prompt,
   and classifies a fresh probe review.

A representative run (gpt-4o-mini):

```
baseline   dev=0.000
 iter 2 [instructions]  dev=1.000 (+1.000)  ACCEPT  — concise, single-word response…
best        dev=1.000
holdout     best=1.000 (baseline=0.000, Δ+1.000) → winner kept
```

The weak prompt scores **0.000** (verbose answers fail strict match); AutoLLM
discovers the "respond with a single word" instruction and reaches **1.000**,
confirmed on the selection-blind holdout.

## Run it

```sh
export OPENAI_API_KEY=sk-...
# Run from the repo root so the run lands in the same local.db the UI reads:
python examples/autollm/agent.py
```

## See it in the UI

The run is persisted (`persist=True`). Launch the Local UI and open **AutoLLM**:

```sh
fastaiagent ui
```

→ **AutoLLM** → the `autollm sentiment demo` run shows the full trajectory
(`baseline → accepted/rejected steps → holdout-guarded winner`) with per-iteration
lever attribution. Each row drills into the **eval run** that scored that
candidate — and from there into the per-case traces. No duplicate eval storage:
the optimize run just links to the eval rows each candidate already produced.

## Notes

- **Algorithm.** AutoLLM uses greedy coordinate ascent (Promptim-style
  keep/revert) with a metaprompt/reflective proposer — the same family as
  LangSmith Promptim and DSPy. The joint-Bayesian variant (DSPy's **MIPRO**) is a
  documented `strategy="mipro"` upgrade path, not the default.
- **Cold-eval scoring** is the OSS path. Replay-grounded scoring (forking a
  production trace at the decision node and rerunning from real operational state)
  is the Enterprise complete-loop capability — the `score_candidate` seam is its
  drop-in point, with no change to the loop driver.
- Two more levers ship in the SDK and are opt-in via `OptimizeConfig(levers=...)`:
  **few-shot** (`"fewshot"`, bootstrapped demos) and **learned-memory**
  (`"memory"`, which subset of learned facts to inject). See
  [docs/evaluation/optimization.md](../../docs/evaluation/optimization.md).
