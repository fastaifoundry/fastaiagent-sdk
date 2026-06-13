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

## Filters

| Filter | Selects traces… | Default expected |
|---|---|---|
| `all` | with any agent span | output-as-expected |
| `favorites` | starred in the Local UI (`trace_favorites`) | output-as-expected |
| `noted` | with a note (`trace_notes`); the note is attached | output-as-expected |
| `guardrail` | where a guardrail fired (`guardrail_events.outcome='fail'`) | needs-review |
| `failed` | with an error-status span (best-effort, see note) | needs-review |

Modifiers: `agent=` (one agent's spans), `since_hours=` (time window),
`limit=` (cap, most-recent first), `dedup_by="input"` (drop duplicate inputs).

!!! note "`failed` is best-effort"
    The agent root span does not always get an `ERROR` status set on exception, so
    `failed` keys off any error-status span in the trace. The **`guardrail`**
    filter is the more reliable failure signal.

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
`limit`, `trace_ids`, `mark_output_as_expected`, `db_path`, `dedup_by` (see
`curate_from_traces`). `Dataset.to_jsonl(path, append=False)` writes the items in
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
`--dedup-by`, `--db`.

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
