# Trace → Dataset Curation

Turn captured agent traces into an eval dataset. Every agent run is already
traced to the local DB; curation reads those traces and emits dataset items
(`{input, expected_output, trace_id, …}`) you can feed straight to `evaluate()`.

This closes the loop: **run agents → curate the interesting traces → evaluate →
improve → repeat.**

## What gets curated — one case per agent span

Curation works at the **agent-span level.** Every `agent.<name>` span becomes one
case via its `agent.input` / `agent.output` attributes — whether that span is a
trace root (a plain `Agent.run`) **or** nested inside a `Chain`, `Supervisor`, or
`Swarm` run. The agent is the core LLM unit, so a 3-agent chain yields 3 cases
(one per agent step), and a supervisor yields its inner-agent case plus one per
worker. Non-agent spans (`llm.*`, `tool.*`, `chain.*` roots) are ignored.

## `expected_output`: good vs. needs-review

The captured output is only a *gold* answer for known-good traces. So curation
branches by intent:

- **Good filters** (`all`, `favorites`, `noted`) → `expected_output` = the
  captured `agent.output` (a "keep producing this" regression case).
- **Failure filters** (`guardrail`, `failed`) → `expected_output = ""` and
  `needs_review = true`; the bad output is kept as `actual_output` with a
  `reason`. Fill in the gold answer before evaluating.

Override per run with `mark_output_as_expected` / `--output-as-expected` /
`--needs-review`.

A dataset curated from `favorites` is safe to [optimize](optimization.md) on: the
few-shot lever also draws demos from your favorites, but never one whose input is a
case the candidates are scored on.

## Infrastructure errors are not gold

A trace can fail for reasons the agent can't fix — endpoint 500, timeout,
network/DB/auth error. Curating such a run as a *gold* case would optimize the
agent against a target it never legitimately produced. So on the good filters,
**a run that produced no usable agent output is dropped, not curated** — the
reliable, agent-attributable signal. A run where a *tool* errored but the agent
recovered and still produced a clean answer is **kept** (that's good signal).

This is controlled by `exclude_infra_errors`:

- `"agent"` *(default)* — drop only when the agent produced no usable output.
- `"trace"` — additionally drop any run whose trace carries an error-status span
  (stricter; also drops tool-errored-but-recovered runs).

The returned dataset reports coverage so a high drop rate doesn't hide silently —
it usually signals an unhealthy agent or a trace-capture problem worth a look,
independent of any optimization built on top:

```python
ds = curate_from_traces(filter="all")
print(ds.coverage_summary())   # "18 case(s) from 220 trace(s); 202 dropped as infra-errored, 0 need review."
ds.infra_excluded              # 202
```

(The `guardrail` / `failed` filters are unaffected — they intentionally surface
bad runs as `needs_review`.)

## Filters

| Filter | Selects traces… | Default expected |
|---|---|---|
| `all` | with any agent span | output-as-expected |
| `favorites` | starred in the Local UI (`trace_favorites`) | output-as-expected |
| `noted` | with a note (`trace_notes`); the note is attached | output-as-expected |
| `guardrail` | where a guardrail fired: a `blocked`, `warned` or `filtered` event (false positives excluded), or, for a run with no event rows, a guardrail span that failed | needs-review |
| `failed` | with an error-status span | needs-review |

Modifiers: `agent=` (one agent's spans), `since_hours=` (time window),
`limit=` (cap, most-recent first), `dedup_by="input"` (drop duplicate inputs),
`include_experiments=True` (see below).

!!! note "Experiments are not curated (1.89.0)"
    Every filter leaves out replay reruns, Playground calls and other traces
    with a non-production `fastaiagent.source`
    ([provenance](../tracing/index.md#where-a-trace-came-from-provenance)).
    Before 1.89.0 a rerun of a failed run was curated as a second failure,
    and its output, which is the rerun's and not the user's, could become a
    gold answer. Pass `include_experiments=True` (CLI `--include-experiments`)
    to curate them anyway. Trace ids passed in `trace_ids=` are never
    filtered.

!!! warning "`guardrail` returned nothing before 1.89.0"
    It matched an event outcome of `fail`, which the SDK has never written, so it
    selected no traces from 1.22.0 to 1.88.0. It now selects any rule that judged
    the payload and fired, including warn, mask, override and reask rules. A
    check that **errored** (could not run) is not selected: that is an
    infrastructure failure, not a quality signal. A firing you marked as a false
    positive in the Local UI is excluded. Event rows exist only for runs made with
    the Local UI enabled; other runs are matched on their guardrail spans, which
    are written whenever tracing is on.

!!! note "`failed` and streamed runs"
    `failed` keys off any error-status span in the trace. Since 1.89.0 a failed
    *streamed* model call also leaves an ERROR span; before, it ended UNSET, so a
    run whose only failure was a streamed call was not selected.

## Python API

```python
from fastaiagent.eval import Dataset

# Curate the traces you starred in the Local UI
ds = Dataset.from_traces(filter="favorites")
ds.to_jsonl("cases.jsonl")

# Re-evaluate against a real agent
from fastaiagent.eval import evaluate
results = evaluate(agent_fn=my_agent.run, dataset="cases.jsonl", scorers=["contains"])
print(results.summary())
```

`Dataset.from_traces(**kwargs)` accepts `filter`, `agent`, `since_hours`,
`limit`, `trace_ids`, `mark_output_as_expected`, `db_path`, `dedup_by`,
`exclude_infra_errors`, `include_experiments` (see `curate_from_traces`). `Dataset.to_jsonl(path, append=False)` writes the items in
the same line format as `ReplayResult.save_as_test`, so curated and
replay-saved cases interleave in one file.

Curating failure traces (need a gold answer before scoring):

```python
ds = Dataset.from_traces(filter="guardrail")
for item in ds:
    if item.get("needs_review"):
        print(item["input"], "->", item["reason"])   # fill in expected_output
```

## CLI

```bash
fastaiagent eval curate --filter favorites --out cases.jsonl
fastaiagent eval curate --filter guardrail --agent support --since 24 --out fixme.jsonl
fastaiagent eval curate --filter all --dedup-by input --out all.jsonl
```

Flags: `--out/-o` (required), `--filter/-f`, `--agent`, `--since` (hours),
`--limit`, `--append/--no-append`, `--output-as-expected/--needs-review`,
`--dedup-by`, `--db`, `--include-experiments`.

## Item shape

Good case:
```json
{"input": "What is the refund window?", "expected_output": "Refunds within 30 days.",
 "trace_id": "0af1…", "source_trace_id": "0af1…", "span_id": "9c…",
 "agent_name": "support", "source": "curated:favorites", "created_at": "…"}
```
Needs-review case:
```json
{"input": "Cancel order #X", "expected_output": "", "needs_review": true,
 "actual_output": "email: a@b.com", "reason": "guardrail 'no_pii' fired: PII detected",
 "trace_id": "9bc2…", "source_trace_id": "9bc2…", "agent_name": "support",
 "source": "curated:guardrail", "created_at": "…"}
```

`evaluate()` reads `input` + `expected_output`; the extra keys are ignored and
preserved for provenance.

See `examples/80_curate_from_traces.py` for an end-to-end runnable script.
