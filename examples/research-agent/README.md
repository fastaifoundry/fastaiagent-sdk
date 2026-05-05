# Research Agent (multi-agent)

A Perplexity / OpenAI-Deep-Research-style research agent built with [FastAIAgent SDK](https://github.com/fastaifoundry/fastaiagent-sdk) v1.6.0. A `Supervisor` coordinates three workers — **researcher**, **writer**, **verifier** — and the verifier has real authority to send the writer back for revisions.

```
                       ┌──────────────────────────────┐
                       │ Supervisor: research-team    │
                       └──────────────┬───────────────┘
                                      │
        ┌─────────────────────────────┼──────────────────────────────┐
        ▼                             ▼                              ▼
   ┌──────────┐                  ┌─────────┐                   ┌──────────┐
   │researcher│                  │ writer  │ ◀── revise ───────│ verifier │
   │ (search) │ ──── findings ──▶│ (draft) │                   │ (audit)  │
   └──────────┘                  └────┬────┘                   └────┬─────┘
                                      │                             │
                                      └──────────── draft ──────────┘
```

**What this example demonstrates** (vs. `examples/customer-support-agent/`):

- `Supervisor` + `Worker` topology with a real revision loop driven by the verifier
- `RunContext[ResearchDeps]` shared across all three workers (the researcher writes the search trail; the verifier reads it later)
- Pluggable backend behind one tool — mock by default; swap in Tavily / Brave / Serper with one env var
- Custom `Scorer` (`RequiredSourcesScorer`) alongside built-in `Faithfulness` + `AnswerRelevancy`
- `Supervisor.astream()` with handoff visibility — see `▷ delegate_to_<role>` events stream live
- Replay across handoff boundaries — the Replay tree is taller because each delegation adds a sub-tree

---

## Quick Start

```bash
# from the SDK root
pip install -e .
cd examples/research-agent
cp .env.example .env        # only OPENAI_API_KEY is required
pip install -r requirements.txt

python agent.py                                  # default topic
python agent.py --topic "Constitutional AI"
```

Ships ready to run offline — the `web_search` tool returns curated mock results for the topics in `eval_suite.py`. Plug in a real backend (Tavily / Brave / Serper) by editing `tools.py` and setting `SEARCH_BACKEND=` in `.env`.

---

## Files

```
research-agent/
├── README.md
├── .env.example
├── requirements.txt
├── tools.py             # web_search FunctionTool + ResearchDeps + real-backend stubs
├── topology.py          # Supervisor + 3 Worker Agents + system prompts
├── agent.py             # CLI entry point
├── streaming_demo.py    # supervisor.astream() with handoff event prints
├── replay_demo.py       # fa.Replay.fork_at(...).rerun() across handoff
└── eval_suite.py        # AnswerRelevancy + Faithfulness + RequiredSourcesScorer
```

---

## How it's wired

### Supervisor + Workers ([topology.py](topology.py))

```python
import fastaiagent as fa
from fastaiagent.agent.middleware import ToolBudget

researcher = fa.Agent(
    name="researcher",
    system_prompt=RESEARCHER_PROMPT,
    llm=llm,
    tools=[web_search],
    middleware=[ToolBudget(max_calls=3)],   # cap searches per delegation
)

writer   = fa.Agent(name="writer",   system_prompt=WRITER_PROMPT,   llm=llm)
verifier = fa.Agent(name="verifier", system_prompt=VERIFIER_PROMPT, llm=llm)

supervisor = fa.Supervisor(
    name="research-team",
    llm=llm,
    workers=[
        fa.Worker(agent=researcher, role="researcher",
                  description="Gathers sources via web_search."),
        fa.Worker(agent=writer,     role="writer",
                  description="Drafts a Markdown report with inline citations."),
        fa.Worker(agent=verifier,   role="verifier",
                  description="Audits draft; returns APPROVED or REVISIONS_REQUESTED."),
    ],
    system_prompt=SUPERVISOR_PROMPT,        # tells supervisor to re-delegate on REVISIONS_REQUESTED
    max_delegation_rounds=8,                # research + write + verify + (write + verify)×2
)
```

### The revision loop

The supervisor's system prompt is the loop logic. It tells the supervisor LLM:

> 1. Delegate to `researcher` to gather sources.
> 2. Delegate to `writer` with topic + findings.
> 3. Delegate to `verifier` with findings + draft.
> 4. **If verifier returns `REVISIONS_REQUESTED:`, delegate back to `writer` with the verifier's specific feedback.** Then re-delegate to `verifier`. Up to 2 revisions.
> 5. Once `APPROVED`, return the writer's final report verbatim.

The verifier's authority comes from the `REVISIONS_REQUESTED:` contract. The supervisor's LLM reads the verifier's tool result, sees the prefix, and re-delegates to the writer with the feedback inlined — no special framework support needed beyond `max_delegation_rounds`.

### Pluggable web search ([tools.py](tools.py))

```python
@fa.tool()
def web_search(query: str, ctx: fa.RunContext[ResearchDeps]) -> str:
    """Search the web for sources on the given query."""
    backend = ctx.state.backend                 # "mock" | "tavily" | "brave" | "serper"
    fn = _BACKENDS.get(backend, _mock_search)
    results = fn(query, ctx.state.top_k)
    for r in results:
        if r not in ctx.state.trail:            # audit trail for the verifier
            ctx.state.trail.append(r)
    return json.dumps(results, indent=2)
```

Three real-backend stubs (`_real_search_tavily`, `_real_search_brave`, `_real_search_serper`) are included with the exact provider URLs and field mappings — fill in the body, set `SEARCH_BACKEND=tavily` in `.env`, and the rest of the example doesn't change.

### Custom scorer ([eval_suite.py](eval_suite.py))

```python
from fastaiagent.eval.scorer import Scorer, ScorerResult

class RequiredSourcesScorer(Scorer):
    name = "required_sources"

    def __init__(self, required_for_case: dict[str, list[str]]):
        self.required_for_case = required_for_case

    def score(self, input, output, expected=None, **kw) -> ScorerResult:
        required = self.required_for_case.get(input, [])
        if not required:
            return ScorerResult(score=1.0, passed=True, reason="no required sources")
        found = [u for u in required if u in output]
        ratio = len(found) / len(required)
        return ScorerResult(score=ratio, passed=ratio >= 1.0,
                            reason=f"{len(found)}/{len(required)} required URLs")
```

This guards against the verifier-loop merely producing a *cited* report that nonetheless missed the canonical paper. Per-case data lives in the dataset; the scorer reads the expected URL set by `input` topic.

---

## Running each entry point

```bash
# Single research run (default topic)
python agent.py
python agent.py --topic "Agent eval"

# Stream the run with handoff visibility
python streaming_demo.py --topic "Transformer architecture"

# Replay debugging — fork at step 1, rerun with a different topic, compare
python replay_demo.py

# Eval suite: AnswerRelevancy + Faithfulness + RequiredSources
python eval_suite.py
python eval_suite.py --publish
```

---

## Local UI

```bash
fastaiagent ui                    # then open http://localhost:8765
```

In the UI:

- `/traces` — the supervisor's trace tree shows `research-team` at the root with three nested worker sub-trees per delegation. Tool calls inside the researcher are visible at the leaf level.
- `/agents` — the dependency graph renders the four-agent topology (supervisor + three workers).
- `/evals` — `eval_suite.py` runs are persisted with per-case scores.
- `/playground` — none registered by default; if you want to externalize the supervisor or worker prompts, register them via `fa.PromptRegistry` (see `customer-support-agent/agent.py` for that pattern) and edit them live here.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `LLM_MODEL` | No | Override the default `gpt-4o` |
| `SEARCH_BACKEND` | No | `mock` (default) / `tavily` / `brave` / `serper` |
| `SEARCH_TOP_K` | No | Results per query (default `4`) |
| `TAVILY_API_KEY` / `BRAVE_SEARCH_API_KEY` / `SERPER_API_KEY` | No | Provider credentials when you swap off mock |
| `FASTAIAGENT_API_KEY` | No | Platform API key for `fa.connect()` |

---

## Customising

**Add a fourth worker** (e.g., a `summarizer` for executive briefings):

```python
summarizer = fa.Agent(name="summarizer", system_prompt="...", llm=llm)
supervisor = fa.Supervisor(
    workers=[..., fa.Worker(agent=summarizer, role="summarizer", description="...")],
    ...
)
```

…then tell the supervisor about it in `SUPERVISOR_PROMPT`.

**Replace the mock search** with the real Tavily API:

```python
def _real_search_tavily(query, top_k):
    import httpx
    r = httpx.post("https://api.tavily.com/search", json={
        "api_key": os.environ["TAVILY_API_KEY"],
        "query": query, "max_results": top_k,
    })
    return [
        {"title": h["title"], "url": h["url"], "snippet": h["content"]}
        for h in r.json()["results"]
    ]
```

…then set `SEARCH_BACKEND=tavily` in `.env`.

**Tighten the verifier**: edit `VERIFIER_PROMPT` to also enforce style (e.g., "every paragraph must end with a citation"). The supervisor will keep looping until the writer satisfies the new constraint or you hit `max_delegation_rounds`.

---

## What this example does NOT demonstrate

- **HITL approval** (`interrupt()` + `aresume()`) — see `customer-support-agent`.
- **Multi-turn memory** in a chat REPL — see `customer-support-agent`.
- **Multimodal** input — workers here are text-only by default; pass an `fa.Image` to `supervisor.arun([prompt, image])` to test it.
- **Chain workflow** with explicit conditional edges — `Supervisor` is the right primitive when you want LLM-driven orchestration; use `Chain` when you want deterministic flow control.

---

## License

Apache 2.0 — same as the SDK.
