# PlatformKB — Hosted Knowledge Bases

`PlatformKB` retrieves from a Knowledge Base hosted on the FastAIAgent platform. The platform runs the full retrieval pipeline (hybrid search, reranking, relevance gate) — the SDK is a thin HTTP client. No local indexes, no embedders, no storage on the agent runtime.

Use this when your KB is managed centrally (uploaded via the platform UI, shared across agents, metered, or regulated) and agents just need to query it at runtime.

## Prerequisites

1. A KB created on the platform with ingested documents (status `ready`).
2. An API key with the `kb:read` scope and domain access to the KB.
3. `fa.connect(...)` called once per process.

## Quick Start

```python
import fastaiagent as fa

fa.connect(api_key="fa_k_...", target="https://<label>-dev.api.fastaiagent.net")

kb = fa.PlatformKB(kb_id="c876a247-4fa1-4796-9d89-cc9a9e5fd4a3")

results = kb.search("refund policy", top_k=3)
for r in results:
    print(f"[{r.score:.3f}] {r.chunk.content[:80]}...")
```

Async variant:

```python
results = await kb.asearch("refund policy", top_k=3)
```

## Using with an Agent

Wire it into an `Agent` via `.as_tool()` — identical to `LocalKB`. Give it a `name` and a
`description` (1.99.0) so the agent knows what it holds:

```python
kb = fa.PlatformKB(
    kb_id="c876a247-4fa1-4796-9d89-cc9a9e5fd4a3",
    name="refund-policy",
    description="refund windows, fees and refund codes",
)
# tool: search_refund-policy — "Search the 'refund-policy' platform knowledge base:
#                               refund windows, fees and refund codes"

agent = fa.Agent(
    name="policy-bot",
    system_prompt="Use the search tool to answer from the KB.",
    llm=fa.LLMClient(provider="openai", model="gpt-4o-mini"),
    tools=[kb.as_tool()],
)

result = agent.run("What's the refund policy?")
print(result.output)
```

Because both `LocalKB` and `PlatformKB` expose the same `search(query, top_k) -> list[SearchResult]` surface and the same `as_tool()` helper, an Agent wired for one works for the other — swap as you promote a KB from local prototyping to platform hosting. Both tools carry `origin="kb"`, so the Local UI lists them under the agent's knowledge bases (before 1.91.0 a `PlatformKB` tool showed up as a plain function tool).

Without `name`, the tool is `search_<kb_id>` and its description names only the id. An
agent holding its own KB and a shared one then can't tell which holds what. Retrieval
spans carry the `name` as `retrieval.kb_name` and keep the id in `retrieval.kb_id`.

Each hit reaches the model as the **whole chunk**, as `LocalKB`'s do. Before 1.99.0
`PlatformKB.as_tool()` cut every hit to its first 200 characters, so a fact further into a
chunk never reached the model. In a live test the agent answered with none of the facts it
was asked for, and searched again and again for text it was never shown.

## What Runs Where

| Stage | LocalKB | PlatformKB |
|---|---|---|
| Embedding | SDK process | Platform |
| Vector search | SDK process (FAISS) | Platform (Qdrant) |
| Keyword search | SDK process (BM25) | Platform |
| Reranking | — | Platform (if configured) |
| Relevance gate | — | Platform (if configured) |
| Storage | Local SQLite | Platform Postgres |

`PlatformKB` has no knobs for retrieval mode — the KB's platform config decides. That's the point: retrieval policy lives with the KB, not the caller.

## Error Handling

`PlatformKB` raises the standard platform errors from `fastaiagent._internal.errors`:

- `PlatformNotConnectedError` — `fa.connect()` was not called.
- `PlatformAuthError` — missing/invalid key, or the key lacks `kb:read`.
- `PlatformNotFoundError` — `kb_id` doesn't exist or is in another domain.
- `PlatformRateLimitError` / `PlatformTierLimitError` — quota exceeded.
- `PlatformConnectionError` — network / 5xx.

## See Also

- [Example 34 — PlatformKB with an Agent](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/34_platform_kb.py)
- [LocalKB](index.md) — local alternative with the same `.search()` surface.
- [Backends](backends.md) — swap `LocalKB`'s vector/keyword/metadata stores.
