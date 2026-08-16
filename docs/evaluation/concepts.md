# Concepts & Mental Model

This page is the mental model for evaluation — *why* you evaluate, the
*pipeline* every eval follows, the *families* of scorers and when to reach for
each, what they *cost*, and how evaluation closes the loop back into a better
agent. Read it first, then use the reference pages
([LLM-as-Judge](llm-judge.md), [Trajectory](trajectory-scoring.md),
[Session](session-scoring.md), [RAG](rag-metrics.md), [Safety](safety-metrics.md),
and the rest) for depth.

## Why evaluate

A prompt change that looks better on one example can quietly regress ten
others. Evaluation replaces "it seemed fine when I tried it" with a repeatable
measurement: run the agent over a dataset, score every case, and get a number
you can compare across versions. It runs **entirely offline** — no cloud
service required — and persists to the local UI so you can track it over time.

You evaluate at four moments: while **developing** (does this change help?),
before **release** (does it beat the last version on a golden set?), in **CI**
(gate merges on a regression threshold), and against **production traces**
(is quality holding on real traffic?).

## The evaluation pipeline

Every eval — one-off or in CI — is the same five-step pipeline:

```
capture ──▶ curate ──▶ evaluate ──▶ report ──▶ improve
(traces)   (dataset)   (score)     (results)   (harden / optimize)
   ▲                                                 │
   └─────────────── re-run to confirm ◀──────────────┘
```

1. **Capture** — real runs emit traces to `local.db`.
2. **Curate** — turn traces into a dataset with `curate_from_traces(...)`
   (filter by favorites, notes, guardrail hits, failures; optionally mark the
   observed output as `expected`). See [Trace Curation](curation.md).
3. **Evaluate** — `evaluate(agent_fn, dataset, scorers=[...])` runs the agent
   over each case and applies each scorer. It returns `EvalResults`.
4. **Report** — `results.summary()` gives per-scorer average and pass-rate;
   results persist to the UI (`persist=True` by default) and can publish to the
   platform or `compare()` against a prior run.
5. **Improve** — feed failures back: [harden](agent-hardening.md) proposes
   fixes, [optimize](optimization.md) searches prompts/few-shots, and
   [simulate](agent-hardening.md) generates fresh adversarial scenarios. Then
   re-run to confirm.

!!! info "Verified against a live run"
    `evaluate(agent_fn=agent.run, dataset=[...], scorers=["contains", GEval(...)])`
    ran the agent over the dataset and returned a summary with a per-scorer
    average and pass rate for both the code scorer and the live LLM judge —
    confirming the dataset → run → score → report path end to end.

The core call is deliberately small:

```python
results = evaluate(
    agent_fn=my_agent.run,          # anything callable(input) -> output
    dataset="cases.jsonl",          # Dataset | path | list[dict]
    scorers=["exact_match", geval], # strings resolve from the registry; or pass instances
    concurrency=4,                  # cases scored in parallel
    persist=True,                   # write to local.db for the UI
)
```

## The concept of how `evaluate()` works

Under the small call is a simple, deterministic loop:

1. **Resolve scorers.** Each string is looked up in the `BUILTIN_SCORERS`
   registry and instantiated; a `Scorer` instance is used as-is. So
   `"exact_match"` and `ExactMatch()` are the same thing.
2. **Run cases concurrently.** An `asyncio.Semaphore(concurrency)` bounds how
   many cases run at once. For each case it calls `agent_fn(input)` to get the
   output. An `async def` callable is awaited directly; a **synchronous** one is
   run on a worker thread (`asyncio.to_thread`) so it can't block the event loop
   and stall the other cases. Two things follow: a sync `agent_fn` must be
   thread-safe, and frameworks that refuse to run synchronously inside a live
   event loop (CrewAI ≥1.15, for one) work correctly.
3. **Score each case.** Every resolved scorer's `score()` returns a
   `ScorerResult(score, passed, reason)` — a numeric `score` plus a boolean
   `passed` (each scorer decides its own pass condition) and an optional reason.
4. **Roll up.** Per scorer, results aggregate into a `MetricSummary(name,
   avg_score, pass_rate, n)` — `avg_score` is the mean score, `pass_rate` the
   fraction of cases that passed. That's what `summary()` prints.
5. **Persist.** With `persist=True`, the run and per-case rows are written to
   `local.db` so the Local UI can show it and `compare()` can diff runs.

The key idea: a scorer is just a pure function `(output, expected, context) →
ScorerResult`. Code scorers compute that directly; LLM-Judge/G-Eval compute it
by asking a model — G-Eval builds a rubric from your steps + scale (optionally
with Auto-CoT) and parses the model's verdict **fail-closed** (an ambiguous
answer scores as a fail). Same contract either way, which is why you can mix
free and paid scorers in one `scorers=[...]` list.

## Scorer families — and when to reach for each

A scorer maps `(output, expected, context)` to a number. They fall into
families by *what they judge*:

| Family | Judges | Examples | Cost |
|--------|--------|----------|------|
| **Core** | Exact/structural correctness | `exact_match`, `contains`, `json_valid`, `regex_match`, `length_between`, `latency`, `cost_under` | Free (code) |
| **Similarity** | Closeness to a reference answer | `SemanticSimilarity`, `BLEUScore`, `ROUGEScore`, `LevenshteinDistance` | Free–cheap (embeddings cost a little) |
| **LLM-as-Judge** | Open-ended quality against a rubric | `LLMJudge`, `GEval` (steps + rubric + Auto-CoT) | Paid (one LLM call/case) |
| **RAG** | Grounding in retrieved context | `Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, `ContextRecall` | Paid (LLM) |
| **Safety** | Harmful/leaky output | `PIILeakage` (regex), `Toxicity`, `Bias`, `PromptInjection`, `OpenAIModeration` | Mixed (regex free, judges paid) |
| **Trajectory** | The *process* (which tools, what path) | `ToolUsageAccuracy`, `StepEfficiency`, `PathCorrectness`, `CycleEfficiency`, `ToolCallCorrectness` | Free (code over the trace) |
| **Session** | Multi-turn coherence | `ConversationCoherence`, `GoalCompletion`, `KnowledgeRetention`, `RoleAdherence`, `ConversationRelevancy` | Mixed (heuristic or LLM) |
| **Agent metrics** | Task-level judgments | `TaskCompletion`, `Hallucination`, `ReflectionQuality` | Paid (LLM) |

Choose by **what you care about**: exact answer → Core; "close enough" prose →
Similarity; open-ended quality with no single right answer → LLM-Judge/G-Eval;
did it use the right tools in the right order → Trajectory; is a multi-turn
conversation coherent → Session; is a RAG answer grounded → RAG; is the output
safe → Safety.

!!! info "Free vs paid"
    Core, trajectory, and most similarity scorers are pure code — run them
    liberally, including in CI. LLM-Judge, RAG, agent-metric, and the
    LLM-backed safety/session scorers each cost an inference call per case;
    they catch nuance code can't, but budget for them. Mix cheap scorers for
    coverage with a judge or two for the judgment calls.

## Two ways to score: dataset loop vs. direct

- **Dataset loop** — `evaluate(...)` calls `agent_fn` on each case and applies
  the scorers. This is the common path for input/output scorers.
- **Direct** — trajectory and session scorers judge a *process*, so you call
  them directly with the trajectory/turns, e.g.
  `ToolUsageAccuracy().score(actual_trajectory=..., expected_trajectory=...)`.
  See [Trajectory Scoring](trajectory-scoring.md) and [Session Scoring](session-scoring.md).

## Where a score lives: per-trace vs. per-run

A score has to attach to *something*, and there are exactly **two** places it
can go. They are different granularities, not alternatives:

| | Per-trace inline score | Per-run batch score |
|---|---|---|
| **Question it answers** | "How good was *this* interaction?" | "How good is the agent on *this dataset*?" |
| **Shape** | An `EVALUATOR` span on the trace, carrying `evaluation.{name,score,label,explanation,annotator_kind}` | An eval run: `evaluate(...)` → `EvalResults.publish()` |
| **Scale** | `evaluation.score` on **0..1** | Whatever the scorer returns |
| **When** | Live traffic, one turn at a time | Development, pre-release, CI |
| **Cardinality** | One trace, one or more scores | One dataset, one number per scorer |

The distinction that trips people up: **publishing an eval run is not a
per-trace annotation.** An `EvalResults` is a measurement over a dataset — it
doesn't tell you the quality of any particular production trace. If you want a
score sitting on a live trace, that's the `EVALUATOR` span.

The SDK doesn't score inline anywhere in `agent.run` — that's a deliberate
default, because judging every production turn costs an inference call per turn.
When you *do* want it, you score explicitly and attach the result:

```python
import fastaiagent as fa
from opentelemetry import trace

scorer = fa.Scorer.from_platform("correctness")
result = scorer.score(input=question, output=answer)

fa.emit_evaluation(
    trace.get_tracer("my.runtime"),
    name="correctness",
    score=result.score,        # normalized to 0..1 — a 1..5 judge is score/5
    label="pass" if result.passed else "fail",
    explanation=result.reason,
)
```

An out-of-range score is clamped to `[0, 1]` and warned about, so a scale
mistake surfaces in your logs. Full recipe, including the foreign-runtime case:
[Guardrails & evals without the runtime](../integrations/primitives-without-the-runtime.md).

## Evaluation in your workflow

- **Dev / unit test** — assert on a few cases with the [pytest plugin](pytest.md)
  (`@case`, `pytest_dataset`).
- **Pre-release** — run a golden dataset and `compare()` to the last version.
- **CI gate** — run cheap (code) scorers on every PR; fail the build on a
  regression threshold.
- **Production** — curate from live traces and re-evaluate to catch drift.

## Next steps

- [Evaluation reference](index.md) — `evaluate()`, datasets, built-in scorers, custom code scorers, `EvalResults`
- [LLM-as-Judge](llm-judge.md) and G-Eval — open-ended quality scoring
- [Trajectory Scoring](trajectory-scoring.md) · [Session Scoring](session-scoring.md) — process and multi-turn
- [RAG Metrics](rag-metrics.md) · [Safety Metrics](safety-metrics.md) · [Similarity Metrics](similarity-metrics.md)
- [Trace Curation](curation.md) — turn traces into datasets
- [Agent Hardening & Scorecard](agent-hardening.md) · [AutoLLM optimization](optimization.md) — close the loop
