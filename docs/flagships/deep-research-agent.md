# Deep Research Agent

A flagship template for long-horizon research, native to fastaiagent-sdk. Implements the now-canonical pattern (popularized by Open Deep Research and Anthropic's research agents): **Scope → parallel Research → Write**.

The full source lives at [`examples/deep-research-agent/`](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/examples/deep-research-agent).

## Architecture

```
ScopeAgent ──→ ResearchBrief
                 │
                 ▼
            Researcher × N  (parallel via asyncio.gather)
                 │
                 ▼
           ResearchFindings × N
                 │
                 ▼
             WriteAgent ──→ Markdown report
```

**Why this shape.** Empirically, multi-agent synthesis fails at writing — outputs become disjoint sections that don't compose. The pattern that works is to *parallelize information gathering* and *serialize writing*. Each sub-researcher works in its own context window on a single subtopic; a single one-shot writer composes the final report from their pruned findings.

## Three phases

### Phase 1 — Scope

A single `ScopeAgent` with `output_type=ResearchBrief` (Pydantic) reads the user's topic and produces a structured brief: a refined topic statement, a 2–3 sentence summary, and 2–5 *independent* subtopics.

```python
class Subtopic(BaseModel):
    title: str
    rationale: str

class ResearchBrief(BaseModel):
    topic: str
    summary: str
    subtopics: list[Subtopic]
```

The brief is persisted on a `deep_research.scope` span as JSON in the `fastaiagent.research.brief` attribute.

### Phase 2 — parallel Research

For each subtopic, the template constructs a fresh `Researcher` agent (its own context window, its own tool budget) and dispatches them all via `asyncio.gather`:

```python
tasks = [
    _run_one_researcher(st.title, st.rationale, ctx)
    for st in brief.subtopics
]
all_findings = await asyncio.gather(*tasks)
```

Each researcher has two tools: `web_search` (Tavily / Brave / Serper / mock fallback) and `web_fetch` (httpx + stdlib HTML stripper). It returns a typed `ResearchFindings` (subtopic, summary, citations).

> **Why no Supervisor here?** The SDK's executor runs tool calls within a single agent turn sequentially today, so going through `Supervisor`'s `delegate_to_<role>` mechanism wouldn't actually parallelize the branches. Cross-agent parallelism via `asyncio.gather` over independent `Agent.arun` calls is the right pattern for this template.

Each branch emits its own `deep_research.research` span carrying the subtopic and structured findings.

### Phase 3 — Write

A single `WriteAgent` receives the brief plus all `ResearchFindings` and composes the final Markdown report in one LLM call. No iteration, no revision loop — the eval suite checks citation coverage post-hoc.

The `deep_research.write` span carries lightweight report metadata (length, citation count).

## Trace shape

```
deep_research.session         ← template.kind="deep-research", topic, plan
  ├── deep_research.scope     ← ResearchBrief (JSON)
  ├── deep_research.research  ← ResearchFindings (parallel × N)
  └── deep_research.write     ← report metadata
```

### Identifying a deep-research trace

The session span carries `fastaiagent.template.kind = "deep-research"`. This is the canonical marker — the UI can render a per-template badge and filter trace lists without parsing span names. Any other flagship template can do the same via `set_template_kind(span, "<kind>")` from `fastaiagent.trace.span`.

Inspect via the local UI or directly:

```sh
# Find every deep-research run by template marker
sqlite3 .fastaiagent/local.db \
  "SELECT trace_id, json_extract(attributes, '\$.fastaiagent.research.topic')
   FROM spans
   WHERE json_extract(attributes, '\$.fastaiagent.template.kind') = 'deep-research';"

# Same listing via the REST API
curl -s 'http://127.0.0.1:7843/api/traces?last_hours=24' \
  | jq '.rows[] | select(.name == "deep_research.session")'

# Drill into one trace's structured payloads
curl -s 'http://127.0.0.1:7843/api/traces/<trace_id>' \
  | jq '.spans[] | select(.name | startswith("deep_research")) | {name, attrs: .attributes}'
```

## Running

```sh
cd examples/deep-research-agent
pip install -r requirements.txt
export TAVILY_API_KEY=...               # optional — falls back to mock
python agent.py --topic "Current state of MCP adoption in 2026"
```

The companion scripts:
- `streaming_demo.py` — per-phase progress trace with timings
- `replay_demo.py` — re-render the most recent run from `local.db`
- `eval_suite.py` — golden questions with two non-LLM scorers

## Configuration

| Env var | Default | Controls |
|---|---|---|
| `LLM_MODEL_SCOPE`      | `gpt-4o`     | Scope agent model |
| `LLM_MODEL_RESEARCHER` | `gpt-4o-mini` | Researcher model (cheap, volume) |
| `LLM_MODEL_WRITER`     | `gpt-4o`     | Writer model |
| `SEARCH_BACKEND`       | `auto`       | `tavily` / `brave` / `serper` / `mock` |
| `SEARCH_TOP_K`         | `5`          | Results per search call |
| `RESEARCH_TOOL_BUDGET` | `6`          | Max tool calls per branch |
| `FETCH_MAX_CHARS`      | `8000`       | Truncate `web_fetch` output |

## What's coming next

`memory_setup.py` is a placeholder for the **trace learning loop** — an offline `fastaiagent learn` CLI that reads completed traces from `local.db`, extracts durable facts, and re-injects them into future runs via `PersistentFactBlock`. Together: a self-improving research agent running entirely on your local machine.
