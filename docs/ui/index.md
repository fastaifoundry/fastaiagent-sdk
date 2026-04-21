# Local UI

A polished, single-user web UI for traces, eval runs, prompts, guardrail events,
and agents — shipped inside the `fastaiagent` wheel. Runs on your laptop, reads
from `./.fastaiagent/local.db`, nothing leaves the machine.

!!! tip "Your project, your UI"
    Zero Docker. Zero Postgres. Zero cloud account.
    `pip install 'fastaiagent[ui]'`, run `fastaiagent ui`, done.

## Install

The UI's web stack (FastAPI, uvicorn, aiosqlite, bcrypt, itsdangerous) lives
behind an optional extra so non-UI users don't pay for it.

```bash
pip install 'fastaiagent[ui]'
```

## First run

```bash
fastaiagent ui
```

First launch prompts for a username and password, saves a bcrypt-hashed
credential to `./.fastaiagent/auth.json`, and opens your browser on
`http://127.0.0.1:7842`.

```text
FastAIAgent Local UI — first run
Set a username: upendra
Set a password: ***
Confirm password: ***
✓ Credentials saved to ./.fastaiagent/auth.json
Starting UI on http://127.0.0.1:7842
Opening browser...
```

![Login page](screenshots/12-login.png)

## Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--host` | `127.0.0.1` | Bind address. Keep on loopback unless you really need LAN access. |
| `--port` | `7842` | Pick anything you like. |
| `--no-auth` | off | Skip login entirely. Intended for throwaway containers, not everyday use. |
| `--no-open` | off | Don't launch the browser. |
| `--db PATH` | `./.fastaiagent/local.db` | Override the local DB path. Also settable via `FASTAIAGENT_LOCAL_DB`. |
| `--auth-file PATH` | `./.fastaiagent/auth.json` | Override the credentials file. |

### Forgot password

```bash
fastaiagent ui reset-password
```

Deletes `./.fastaiagent/auth.json`. Next `fastaiagent ui` prompts you to
create new credentials.

---

## Tour

Screenshots below are captured from a real running instance against the
seeded snapshot DB — they stay in sync with the code via
`scripts/capture-ui-screenshots.sh`.

### Home

The overview lands you on "what happened since I last looked": traces in the
last 24 hours, failing traces, eval runs in the last 7 days, and average
pass rate. Two side-panels list the most recent traces and eval runs so you
can jump straight in.

![Home overview](screenshots/01-overview.png)

### Traces

Compact, monospace-numeric list with filters on top: search across
name/input/output, time-range pills (15m / 1h / 24h / 7d / All), status
selector, **runner-type pill (Agent / Chain / Swarm / Supervisor)**, agent
name, and thread id. Every row carries a **Workflow** badge that tells you
at a glance whether the trace was a single agent or a multi-agent
orchestration. Per-row copy-trace-id, favorite, and delete buttons. Click
any row to open the detail view; multi-select + bulk delete available
from a sticky toolbar.

![Traces list](screenshots/02-traces.png)

#### How workflows are traced

`Agent.arun()` emits an `agent.<name>` root span. When you run a
**Chain**, **Swarm**, or **Supervisor**, the SDK wraps the whole run in
one `chain.<name>` / `swarm.<name>` / `supervisor.<name>` root span, and
every child agent + LLM call nests beneath it. That means a 3-agent
chain is **one** trace with a tree, not three orphan agent traces — and
the Workflow badge shows you which kind of runner it was.

Everything the SDK does is traced as a span in that tree:

| Span name | Emitted by | Notable attributes |
|---|---|---|
| `agent.<name>` | `Agent.arun()` | `agent.name`, `agent.input`, `agent.output`, `agent.tokens_used`, `agent.latency_ms`, `agent.llm.*` |
| `chain.<name>` / `swarm.<name>` / `supervisor.<name>` | `Chain.execute()` / `Swarm.arun()` / `Supervisor.arun()` | `fastaiagent.runner.type`, `chain.node_count`, `swarm.entrypoint`, etc. |
| `llm.<provider>.<model>` | `LLMClient.complete()` | `gen_ai.request.*`, `gen_ai.usage.*`, `gen_ai.response.*` |
| `tool.<name>` | every `@tool` / `FunctionTool.aexecute` | `tool.name`, `tool.args`, `tool.status`, `tool.result`, `tool.error` |
| `retrieval.<kb_name>` | `LocalKB.search()` / `PlatformKB.search()` | `retrieval.backend`, `retrieval.search_type`, `retrieval.query`, `retrieval.top_k`, `retrieval.result_count`, `retrieval.latency_ms`, `retrieval.doc_ids` |

The Inspector's **Input** tab surfaces whichever of `*.input` / `tool.args` /
`retrieval.query` is present on the selected span; **Output** surfaces
`*.output` / `tool.result` / `retrieval.doc_ids` / `gen_ai.response.*`.
Payload-bearing attributes (messages, queries, doc ids) respect
`FASTAIAGENT_TRACE_PAYLOADS=0` if you want structural-only tracing.

### Trace detail

Summary bar across the top with trace id, agent, duration, span count,
tokens, cost, and status pill. The left pane is a Gantt-style span tree —
icons and colors per span type (agent / LLM / tool / retrieval / guardrail),
indentation reflects the parent→child relationship, error spans are marked.
The right pane is an inspector with four tabs (**Input / Output /
Attributes / Events**); each renders as a JSON viewer with copy-on-hover.

![Trace detail](screenshots/03-trace-detail.png)

### Agent Replay

The same span tree as Trace Detail, but with a **Fork here** button in the
header. Pick a span on the tree, open the fork dialog.

![Agent Replay](screenshots/04-agent-replay.png)

The fork dialog has four tabs for the four kinds of modification:

- **Prompt** — override the system prompt at the forked step.
- **Input** — provide a new input JSON at this span.
- **Tool response** — inject a canned tool return value.
- **LLM params** — change temperature / max tokens.

![Fork dialog](screenshots/05-replay-fork-dialog.png)

After rerun completes, a side-by-side comparison panel appears below with
the original vs. new output and a step-by-step comparison of both traces,
highlighting where they diverged. A **Save as regression test** button
appends the case to `./.fastaiagent/regression_tests.jsonl` so
`evaluate()` can pick it up.

### Eval runs

A pass-rate trend chart at the top (runs over time, grouped by dataset) plus
a table of every run with dataset, scorers, pass-rate bar, and started-ago.

![Eval runs](screenshots/06-evals.png)

Click a run to see per-case results. Each case shows input / expected /
actual plus a pass/fail chip per scorer. Click the ▶ icon on any row to
open that case's trace in Replay.

![Eval run detail](screenshots/07-eval-detail.png)

### Prompts

Registry browser — list every prompt with latest version, total versions,
and the number of traces that used it. Click to edit.

![Prompts list](screenshots/08-prompts.png)

The editor lists versions on the left, with the template on the right
(auto-detected `{{variable}}` placeholders shown in the header). Save
creates a new version; the lineage panel below lists every trace and eval
run using this prompt.

When the registry lives outside the current project folder the editor is
disabled and a banner explains why (the rule is "UI mutates only what's
clearly local and personal"; external paths are owned by whoever runs that
environment).

![Prompt editor](screenshots/09-prompt-editor.png)

### Guardrail events

Every guardrail firing — name, type, position, outcome pill (passed /
blocked / warned), score, agent, message. Filter by rule / outcome / agent.
Click the ↗ icon to jump to the parent trace.

![Guardrail events](screenshots/10-guardrails.png)

### Agents

Cards summarizing every agent the SDK has seen: run count, success rate
(color-graded), average latency, average cost, last-run time. Click a card
to see the full trace list filtered to that agent.

![Agents](screenshots/11-agents.png)

### Analytics

Latency percentiles (p50 / p95 / p99), cost over time, error rate, and
trace volume charts across a configurable window (24h / 7d / 30d). Below,
top-5 slowest agents and top-5 priciest agents — Langfuse-style signals
that tell you where to invest performance or cost work.

![Analytics](screenshots/13-analytics.png)

### Thread view

Agent runs that share the same `thread_id` span attribute group into a
thread (equivalent to a "session" in Langfuse). Open one from the **Thread**
column on the Traces list, from the pill on a Trace Detail summary bar, or
by hitting `/threads/<id>` directly.

![Thread view](screenshots/14-thread.png)

### Scores on a trace

The Trace Detail page now shows every score attached to the trace: each
guardrail event (passed / blocked / warned) and every eval case that
pointed at this trace. Click through to the owning eval run.

![Trace scores](screenshots/15-trace-scores.png)

### Knowledge Bases (read-only)

**Sidebar → Knowledge Bases**. Every `LocalKB` collection found under
`./.fastaiagent/kb/` (or `$FASTAIAGENT_KB_DIR`) appears with its document
count, chunk count, size on disk, and last-updated timestamp.

![Knowledge Bases list](screenshots/16-kb-list.png)

Open a collection to get three tabs:

- **Documents** — every ingested source with chunk count and preview; click
  one to see its chunks inline.

  ![KB documents tab](screenshots/17-kb-documents.png)

- **Search playground** — type a query, pick a `top_k`, click **Run**. The
  UI calls the same `kb.search()` you'd use from code and shows ranked
  chunks with similarity scores and metadata. No streaming — one
  request, one set of results, user clicks **Refresh** for more.

  ![KB search playground](screenshots/18-kb-search.png)

- **Lineage** — agents and recent traces that issued `retrieval.<kb>`
  spans, derived from the spans table. Great for answering "who's
  actually hitting this KB and when?"

  ![KB lineage tab](screenshots/19-kb-lineage.png)

The UI never writes to a KB. Adding, deleting, or re-indexing documents
stays in code (`kb.add()`, `kb.delete()`, `kb.clear()`) — the Local UI
is a read-only browser, consistent with the rest of Local tier.

See [KB browser →](kb.md) for the full tour.

---

## Managing disk space

Traces add up. Two ways to clean up:

- **Per-row**: trash icon on any row of `/traces`, with a confirmation
  dialog that lists exactly what will be removed (spans, notes, favorites,
  and linked guardrail events — eval cases are kept with a nulled
  `trace_id`).
- **Bulk**: select checkboxes on the left of `/traces` and click
  **Delete N** in the sticky bulk-action toolbar.

Or at the filesystem level, `rm .fastaiagent/local.db` nukes everything
local and `fastaiagent ui` starts fresh.

---

## Data

Everything lives in a single SQLite file at `./.fastaiagent/local.db`:

| Category | Tables |
|---|---|
| Traces | `spans` |
| Checkpoints | `checkpoints` |
| Prompts | `prompts`, `prompt_versions`, `prompt_aliases`, `prompt_fragments` |
| Evals | `eval_runs`, `eval_cases` |
| Guardrails | `guardrail_events` |
| UI view-state | `trace_notes`, `trace_favorites`, `saved_filters` |

No cloud dependency. No external service. Copy the file, back it up, or
`rm .fastaiagent/local.db` to start fresh.

## Migration from 0.7.x

0.7.x wrote three locations: `.fastaiagent/traces.db`, `.fastaiagent/checkpoints.db`,
and `./.prompts/*.yaml`. 0.8 unifies them into `./.fastaiagent/local.db`.

```bash
fastaiagent migrate
```

Copies spans, checkpoints, prompts, and fragments from the legacy stores
into `local.db`. Idempotent — safe to run multiple times. Legacy files are
left in place; delete them once you've confirmed the report.

`fastaiagent ui start` invokes `migrate` automatically when it notices
legacy files on first launch.

## Architecture

The UI is a FastAPI server plus a static React SPA:

```
fastaiagent ui  ──►  FastAPI (uvicorn)  ──►  local.db (SQLite)
                      │                      ▲
                      ▼                      │
                    static/index.html        │ writes
                    + assets/                │
                                            agent runs,
                                            guardrail execs,
                                            evaluate() calls
```

The frontend is a plain React 19 + Vite SPA built at release time and bundled
into the wheel under `fastaiagent/ui/static/`. At runtime, FastAPI serves the
bundle and an `/api/*` REST surface. **There is no WebSocket or live stream** —
every page refreshes on user action via React Query.

## Privacy

- Binds to `127.0.0.1` by default — nothing on your LAN can reach it.
- `HttpOnly` + `SameSite=Strict` session cookie.
- No telemetry. No phone-home. No account.
- `--no-auth` is available for throwaway containers but NOT the default.

## Testing

The UI ships with a full test pyramid:

- **Backend (pytest)** — `tests/test_ui_server.py` exercises every REST
  route against a real FastAPI app + real SQLite fixtures + real bcrypt
  auth. `tests/test_ui_events.py`, `tests/test_ui_cli.py`,
  `tests/test_ui_migration.py`, `tests/test_ui_db.py` cover the rest.
- **Frontend unit (Vitest + Testing Library)** — real DOM rendering
  through the Provider stack (`src/test/utils.tsx`). Coverage includes
  format helpers, status badges, pass-rate bar, sidebar routing, traces
  table, span tree interactions, and the login flow.
- **Frontend E2E / screenshots (Playwright)** —
  `ui-frontend/tests/screenshots.spec.ts` drives a real browser against the
  FastAPI server and captures the screenshots shown above. Run it with:

  ```bash
  bash scripts/capture-ui-screenshots.sh
  ```

  The script seeds a snapshot DB, starts the server on `127.0.0.1:7843` in
  `--no-auth` mode, runs every screenshot test, and tears down.

All three layers run against real libraries (real SQLite, real FastAPI,
real browser, real bcrypt) — no mocking of the subject under test.
