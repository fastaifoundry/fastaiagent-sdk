# Deep Research Agent

A flagship template for **long-horizon research** on the fastaiagent-sdk runtime. Implements the pattern popularized by LangChain's *Open Deep Research* and Anthropic's research agents:

```
ScopeAgent ──→ ResearchBrief
                 │
                 ▼
            Researcher × N  (parallel)
                 │
                 ▼
           ResearchFindings × N
                 │
                 ▼
             WriteAgent ──→ Markdown report
```

**Why this shape:** parallelize information gathering, serialize writing. Multi-agent synthesis tends to produce disjoint sections; a single one-shot writer keeps the report coherent.

## Quick start

```sh
# 1. Install (uses the existing fastaiagent install)
pip install -r requirements.txt

# 2. (Optional) plug in real web search
export TAVILY_API_KEY=...   # or BRAVE_SEARCH_API_KEY / SERPER_API_KEY
export SEARCH_BACKEND=tavily   # default 'auto' picks tavily if key is set

# 3. Run
python agent.py --topic "Current state of MCP server adoption"
```

If no key is set, it falls back to a small mock corpus so the template runs offline (good for first-look + CI).

## What you get

- **3-phase pipeline** (`agent.py`) — Scope → parallel Research → Write, orchestrated explicitly with `asyncio.gather` over plain `Agent` instances. No Supervisor required; cross-agent parallelism is the template's responsibility.
- **Real web tools** (`tools.py`) — `web_search` (Tavily / Brave / Serper / mock) + `web_fetch` (httpx + stdlib HTML stripper, no extra deps).
- **Structured trace spans** (`spans.py`) — research brief, plan, and findings are persisted as JSON in span attributes under the `fastaiagent.research.*` namespace. Replay tooling can reconstruct them.
- **Streaming + replay demos** — `streaming_demo.py` prints per-phase progress; `replay_demo.py` re-renders the most recent run from `local.db`.
- **Eval suite** (`eval_suite.py`) — golden-question regression with two non-LLM scorers (citation count, citations-in-retrieval-trail).

## Files

| File | Purpose |
|---|---|
| `agent.py`        | Entrypoint + pipeline orchestrator |
| `topology.py`     | Scope / Researcher / Writer agent factories + Pydantic schemas |
| `tools.py`        | `web_search`, `web_fetch`, `DeepResearchDeps` |
| `spans.py`        | Helpers for `fastaiagent.research.*` span attributes |
| `memory_setup.py` | Placeholder for the trace-learning loop (PR B) |
| `eval_suite.py`   | Golden-question regression + custom scorers |
| `streaming_demo.py` | Per-phase progress trace |
| `replay_demo.py`  | Re-render past runs from `local.db` |

## Configuration

| Env var | Default | What it controls |
|---|---|---|
| `LLM_PROVIDER`       | `openai`     | LLM provider for all three phases |
| `LLM_MODEL_SCOPE`    | `gpt-4o`     | Scope agent model (judgment-heavy) |
| `LLM_MODEL_RESEARCHER` | `gpt-4o-mini` | Research worker model (volume-heavy, cheap) |
| `LLM_MODEL_WRITER`   | `gpt-4o`     | Writer model |
| `SEARCH_BACKEND`     | `auto`       | `auto` / `tavily` / `brave` / `serper` / `mock` |
| `SEARCH_TOP_K`       | `5`          | Results per search call |
| `RESEARCH_TOOL_BUDGET` | `6`        | Max tool calls per research branch |
| `FETCH_MAX_CHARS`    | `8000`       | Truncate `web_fetch` output to N chars |

## Observability

Every run emits a span tree:

```
deep_research.session            ← template.kind="deep-research", topic, plan
  ├── deep_research.scope        ← ResearchBrief (structured JSON)
  ├── deep_research.research × N ← ResearchFindings per subtopic (parallel)
  └── deep_research.write        ← report metadata
```

The session span is tagged with `fastaiagent.template.kind = "deep-research"` so the local UI can identify and filter deep-research runs without parsing span names.

Inspect via the local UI (`fastaiagent ui`) or query directly:

```sh
# Find every deep-research run by template marker
sqlite3 .fastaiagent/local.db \
  "SELECT trace_id, json_extract(attributes, '\$.fastaiagent.research.topic')
   FROM spans
   WHERE json_extract(attributes, '\$.fastaiagent.template.kind') = 'deep-research';"

# Or via the REST API
curl -s 'http://127.0.0.1:7843/api/traces?last_hours=24' \
  | jq '.rows[] | select(.name == "deep_research.session")'
```

## What's coming next (PR B)

The `memory_setup.py` file is a placeholder for the **trace learning loop** — an offline `fastaiagent learn` CLI that reads completed traces from `local.db`, extracts durable per-user/per-project facts, and re-injects them into future runs via a `PersistentFactBlock`.

Together, this template + the learning loop = a self-improving research agent running entirely on your local machine.
