# Eval Datasets

Curate eval cases inside the UI instead of hand-editing JSONL.
Datasets are still the same files `Dataset.from_jsonl()` loads — the
editor just replaces the script-edit-rerun loop with point-and-click
CRUD, multimodal upload, import/export, and a one-click "Run eval"
against the registered agents.

Find it in the sidebar under `// EVALUATION → Datasets`, or save your
first case from the Playground via the **Save as eval case** button.

![Datasets list](screenshots/sprint3-5-datasets-list.png)

## What it does

**List page** (`/datasets`) — every `*.jsonl` under
`./.fastaiagent/datasets/` with its case count, last-modified time, and
a multimodal badge if any case has an image or PDF part.

- **New dataset** creates an empty `<name>.jsonl`. Name is restricted to
  `[A-Za-z0-9_-]+` so paths can't escape the datasets directory.
- **Export** downloads the file directly via
  `GET /api/datasets/{name}/export` — no client-side serialization.
- **Delete** removes the JSONL plus the per-dataset image folder
  (`images/{name}/`); persisted eval runs that pointed at this dataset
  keep their snapshots.

![Dataset detail with cases table](screenshots/sprint3-6-dataset-detail.png)

**Detail page** (`/datasets/{name}`) — the cases table with row actions
and a header full of bulk affordances:

- **Add case** opens the editor modal; the new line is appended to the
  JSONL atomically (tmp file → `os.replace`).
- **Import cases** uploads a JSONL file, validating each line has an
  `input` field. The first malformed line wins a 400 with the line
  number; nothing partial is written. `mode=append` (default) preserves
  existing cases; `mode=replace` overwrites.
- **Export** downloads the JSONL.
- **Run eval** opens a dialog (*1.89.0*) that runs the eval framework on the
  dataset:
  - **Target.** A registered agent, or a **dataset check**, which echoes each
    input back: it tests that the dataset loads and scores, not a model, and is
    recorded as `kind: ui-echo` and left out of the trend and the Home
    headline. Agents are selectable once registered: start the UI with
    `fastaiagent ui --agent app.py:my_agent`. A Chain is listed but cannot be
    evaluated (it has no `run()`); evaluate its agents. Before 1.89.0 the
    button always ran the echo check with `exact_match`, and no agent could be
    picked.
  - **Scorers**, grouped as deterministic (`exact_match`, `contains`,
    `regex_match` with its pattern, `length_between`, `latency`,
    `cost_under`, BLEU, ROUGE, Levenshtein, PII, prompt injection), LLM judges
    (`task_completion`, `toxicity`, `bias`, `answer_relevancy`, your own
    `llm_judge` criteria, or `geval` with steps and a rubric, each with a judge
    model chosen from those with a key on the server), trajectory scorers (they
    read the agent's tool calls against each row's `expected_trajectory` /
    `expected_tool_calls` / `expected_steps`) and RAG scorers (each row's
    `context`). A scorer the dataset cannot feed is disabled with the reason.
  - **Warnings** before you run: rows missing what a scorer needs (those cases
    are n/a), image or PDF rows under text scorers, and roughly how many judge
    calls the run makes.
  - The result shows cases passed / failed / errored, the checks rate and a
    link to `/evals/{run_id}`. Every invalid request (an unknown scorer, a bad
    parameter, a regex that does not compile, a judge model with no key, the
    same scorer twice, no target chosen, an image a row points at that cannot
    be loaded) is a `422` that says why, and nothing runs. Judge-backed runs
    share the Playground's per-minute LLM rate limit. Runs are synchronous.
- **Fields the editor does not show are kept** (*1.89.0*). A row's
  `expected_trajectory`, `context`, `source_trace_id` and any other field
  appear read-only under "Other fields". Before 1.89.0 adding, editing or
  deleting any case rewrote the file with only `input`, `expected_output`,
  `tags` and `metadata`, so those fields were dropped from **every** row.

## Case editor

![Case editor with multimodal attachment](screenshots/sprint3-7-case-editor-multimodal.png)

- **Input** — plain textarea. Attach images via the **Attach image**
  button; the upload lands at `<datasets>/images/<name>/<file>` and the
  case input becomes a typed-parts list:
  ```json
  {"input": [
    {"type": "text", "text": "What animal is this?"},
    {"type": "image", "path": "images/vision/cat.png"}
  ], "expected_output": "cat"}
  ```
  This is exactly the shape `Dataset._resolve_multimodal_part` walks at
  load time, so the case is immediately runnable through the eval
  framework.
- **Expected output** — optional. Leave blank for cases scored by
  LLM-as-Judge only (no exact-match assertion).
- **Tags** — comma-separated. Useful for filtering runs by category
  (`exact_match`, `retrieval`, `edge_case`).
- **Metadata** — collapsible JSON object. The eval framework doesn't
  read it, but the field is preserved on disk so the developer can
  attach provenance (e.g. `{"source": "production_trace_abc"}`).

## "Save as eval case" — Playground bridge

![Playground combo over existing datasets](screenshots/sprint3-8-save-as-eval.png)

The Playground's **Save as eval case** button now combos over existing
datasets (with a `+ New` escape hatch), so the inner-loop iteration in
the Playground feeds the outer-loop curation in the editor without
copy-pasting a name. The same dialog is used by Replay's *Save as
regression test*.

## From captured traces (bulk, SDK / CLI)

Beyond the in-UI editor, build datasets programmatically from captured traces —
`Dataset.from_traces(filter="favorites")` or `fastaiagent eval curate --filter
guardrail --out fixme.jsonl`. Each `agent.<name>` span becomes one case. See
[Trace Curation](../evaluation/curation.md).

## JSONL format

Plain text case:

```json
{"input": "Reply with the word 'hello'.", "expected_output": "hello", "tags": ["smoke"], "metadata": {}}
```

Multimodal case:

```json
{"input": [{"type": "text", "text": "What animal is this?"}, {"type": "image", "path": "images/echo/cat.png"}], "expected_output": "cat"}
```

The editor accepts both `expected` and `expected_output` on read (so
hand-edited spec-format files keep working) and writes
`expected_output` on save (matching the Playground's existing
convention so files written by either tool stay consistent).

## Endpoints

```
GET    /api/datasets                          → DatasetSummary[]
POST   /api/datasets                          → 201 DatasetSummary
GET    /api/datasets/{name}                   → DatasetDetail
DELETE /api/datasets/{name}                   → 204
POST   /api/datasets/{name}/cases             → 201 CaseRow (append)
PUT    /api/datasets/{name}/cases/{index}     → 200 CaseRow (in-place)
DELETE /api/datasets/{name}/cases/{index}     → 204 (re-indexes the rest)
POST   /api/datasets/{name}/import            → ImportResult (multipart)
GET    /api/datasets/{name}/export            → JSONL download
POST   /api/datasets/{name}/images            → 201 ImageUploadResult
GET    /api/datasets/{name}/images/{filename} → image bytes
GET    /api/datasets/{name}/run-eval/options  → agents, scorer catalog, judge models, row features
POST   /api/datasets/{name}/run-eval          → RunEvalResult
```

The run-eval body (*1.89.0*):

```json
{
  "agent_name": "support",
  "scorers": [
    "contains",
    {"type": "builtin", "name": "regex_match", "params": {"pattern": "^Order"}},
    {"type": "builtin", "name": "task_completion",
     "model": {"provider": "openai", "model": "gpt-4o-mini"}},
    {"type": "llm_judge", "name": "tone", "criteria": "Polite and brief?",
     "scale": "1-5", "threshold": 0.6,
     "model": {"provider": "openai", "model": "gpt-4o-mini"}}
  ]
}
```

or `{"mode": "echo", "scorers": [...]}` for a dataset check. The specs are
`fastaiagent.eval.scorer_specs`, usable from Python too:
`build_scorer({...})` returns the scorer, validated, without calling a model.
The response adds `kind`, `cases {passed, failed, errored, pass_rate}` and
`checks {passed, failed, errored, skipped, pass_rate}`.

Every endpoint is project-scoped via the standard `AppContext` wiring;
the dataset directory resolves relative to the project's
`.fastaiagent/local.db`, so multi-project setups (Postgres) keep their
cases isolated.

## Integration matrix

| From | Action | Lands in |
|---|---|---|
| Playground | Save as eval case | New row in chosen dataset |
| Replay | Save as regression test | New row in chosen dataset |
| Dataset editor | Add case | Same JSONL the eval framework reads |
| Dataset editor | Run eval | New row in `/evals` |
| Eval Runs | Compare with… | `/evals/compare` (existing) |

Same observability surface throughout — every editor write is
synchronous and the file is the source of truth.

## Example

The example script
[`examples/53_dataset_editor.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/53_dataset_editor.py)
seeds a dataset with text and image cases and prints the
`/datasets/<name>` URL so you can open the editor against real data.
