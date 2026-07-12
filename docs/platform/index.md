# Platform Connection

FastAIAgent SDK works fully standalone. Optionally connect to [FastAIAgent Platform](https://app.fastaiagent.net) for production observability, prompt management, and evaluation services.

## Design Principle

**Local-first, platform-optional.** Every feature works locally with zero platform dependency. `fa.connect()` is an optional upgrade that replaces local backends with platform services. Your code doesn't change — only where data goes and where config comes from.

```python
# Without connect — everything works locally
agent = Agent(name="support", ...)
result = agent.run("Help me")
# Traces → local SQLite
# Prompts → local files
# Eval → local results

# With connect — same code, platform backends
import fastaiagent as fa
fa.connect(api_key="fa-...", project="my-project")
result = agent.run("Help me")
# Traces → platform (with local SQLite fallback)
# Prompts → platform registry (with local cache)
# Eval → results published to platform
```

## Setup

### 1. Create an Account

**SaaS (hosted):**
- Go to [https://app.fastaiagent.net](https://app.fastaiagent.net)
- Sign up with email or SSO

**On-premise (self-hosted):**
- Navigate to your organization's FastAIAgent instance URL

### 2. Create an API Key

- Go to Settings -> API Keys -> Create Key
- Select the **domain** and optionally a specific **project**
- Copy the key (shown only once): `fa-...`

### 3. Connect

```python
import fastaiagent as fa

fa.connect(
    api_key="fa-...",
    target="https://app.fastaiagent.net",  # default
    project="my-project",
)
```

**Environment variables** (alternative):
```bash
export FASTAIAGENT_API_KEY=fa-...
export FASTAIAGENT_TARGET=https://app.fastaiagent.net
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | Required | Platform API key |
| `target` | `str` | `https://app.fastaiagent.net` | Platform URL |
| `project` | `str \| None` | None | Project scope |

## Services

### Observability (Traces)

Every `agent.run()` automatically sends traces to the platform. Local SQLite storage continues as both primary local store and offline fallback.

```python
fa.connect(api_key="fa-...", project="my-project")
result = agent.run("Help me")
# Trace automatically sent to platform
```

What you see on the platform:
- Trace dashboard with all SDK-generated traces
- Span inspection, token counts, cost, latency
- Agent Replay on platform UI
- Per-agent analytics
- Traces from all team members in one view

**Manual backfill** — publish existing local traces:

```python
trace_store = TraceStore()
for t in trace_store.list_traces(limit=100):
    trace_data = trace_store.get_trace(t.trace_id)
    trace_data.publish()
```

### Prompt Registry

Pull versioned, tested, approved prompts from the platform. Non-engineers manage prompts in the UI. Your SDK agents always use the latest deployed version.

```python
registry = PromptRegistry()

# Pull prompt from platform (latest deployed version)
prompt = registry.get("support-prompt")

# Pull specific version
prompt = registry.get("support-prompt", version=3)

# Use in agent
agent = Agent(
    name="support",
    system_prompt=prompt.template,
    ...
)

# Publish a prompt to the platform
registry.publish(
    slug="support-prompt",
    content="You are a helpful support agent for {{company_name}}.",
    variables=["company_name"],
)
```

**Source control:**

```python
prompt = registry.get("support-prompt", source="platform")  # platform only
prompt = registry.get("support-prompt", source="local")      # local only
prompt = registry.get("support-prompt", source="auto")       # platform if connected, else local (default)
```

Platform prompts are cached locally (TTL: 5 minutes). Invalidate with `registry.refresh("support-prompt")`.

### Evaluation

Pull shared datasets, run evals locally, publish results to the platform.

```python
# Pull dataset from platform
dataset = Dataset.from_platform("golden-test-set")

# Run eval locally
results = evaluate(agent, dataset=dataset)

# Publish results to platform
results.publish(run_name="v2.1-release-candidate")

# Push local dataset to platform for team sharing
local_dataset = Dataset.from_jsonl("my_tests.jsonl")
local_dataset.publish("regression-tests")

# Pull scorer config from platform
scorer = Scorer.from_platform("correctness-judge")
```

### Knowledge Bases

Query a KB that was uploaded and ingested on the platform — the platform runs the full retrieval pipeline (hybrid search, reranking, relevance gate). Requires the `kb:read` scope on your API key.

```python
kb = fa.PlatformKB(kb_id="kb_abc123")

results = kb.search("refund policy", top_k=3)
for r in results:
    print(f"[{r.score:.3f}] {r.chunk.content[:80]}...")

# Same wiring as LocalKB — agents don't know which they got.
agent = fa.Agent(
    name="policy-bot",
    llm=fa.LLMClient(provider="openai", model="gpt-4o-mini"),
    tools=[kb.as_tool()],
)
```

See [PlatformKB](../knowledge-base/platform-kb.md) for the full API and platform vs. local trade-offs.

### Replay

Pull any trace from the platform and replay locally:

```python
replay = Replay.from_platform(trace_id="tr-abc123")
replay.step_through()

# Fork from a platform trace
forked = replay.fork_at(step=3)
forked.modify_prompt("Updated system prompt")
result = forked.rerun()
```

### Pushing agent definitions

Agent definitions are **not** auto-synced by `fa.connect()` (agents are code, not config).
When you *do* want a connected control plane to know about an agent — so it appears in the
console with its prompt, model, tools, and memory — serialize it with `Agent.to_dict()` and
`POST` the payload to `/public/v1/sdk/agents` (scope `agent:write`, upsert by `name`):

```python
from fastaiagent._platform.api import get_platform_api

agent = Agent(
    name="support-bot",
    prompt_slug="support-prompt",   # references a governed registry prompt (optional)
    llm=LLMClient(provider="openai", model="gpt-4o"),
    memory=AgentMemory(),           # → memory_enabled: true (optional)
    tools=[...],
)
get_platform_api().post("/public/v1/sdk/agents", agent.to_dict())
```

`to_dict()` emits `{name, agent_type, system_prompt, llm_endpoint, tools, guardrails,
config}` plus two governed fields **only when configured** (an agent with neither is
serialized exactly as before):

| Field | Emitted when | Effect |
|-------|--------------|--------|
| `prompt_slug` | `Agent(prompt_slug=...)` is set | References a governed registry prompt. `system_prompt` is sent as `""` (the slug wins) so the console shows the **slug**, not "Inline". |
| `memory_enabled` | `memory=` is configured | Console shows memory **Enabled**. |

A runnable end-to-end demo (publish prompt → push agent → read back governance) is in
`examples/89_connected_agent_push.py`.

## What Does NOT Flow Through `fa.connect()`

| Capability | Reason |
|-----------|--------|
| Agent definitions (automatically) | Agents are code; push explicitly via `to_dict()` + `POST /public/v1/sdk/agents` (see above) |
| Tool implementations | Tools are Python functions in SDK |
| Guardrail definitions | Guardrails are code-configured in SDK |
| Chain definitions | Chains are code in SDK |
| KB ingestion / document upload | Done on the platform UI or admin API, not via `fa.connect()`. Use [`PlatformKB`](../knowledge-base/platform-kb.md) at runtime to *query* a platform-hosted KB |
| LLM endpoint credentials | SDK manages its own API keys |

## Offline / Disconnected Behavior

Every service degrades gracefully when the platform is unreachable:

| Service | Connected | Disconnected |
|---------|-----------|-------------|
| Traces | Send to platform + local SQLite | Local SQLite only, **buffered for re-send** |
| Prompts | Fetch from platform (cached locally) | Use local cache or local files |
| Eval datasets | Pull from platform | Use local JSONL/CSV |
| Eval results | Publish to platform | Store locally, publish later |
| Replay | Pull platform traces | Local traces only |
| HITL events | Report pause/resolution to platform | Local SQLite only, **buffered for re-send** |
| Central memory | Read curated facts via `PlaneFactBlock` | Block injects nothing; agent runs normally |

No operation fails because the platform is down.

For the connected human-in-the-loop observer (pause/resolution reporting, the
compliance ledger, and the `connected_state_plane` gate), see
[Connected HITL](connected-hitl.md).

For connected central memory — reading curated, human-approved facts back into an
agent via `PlaneFactBlock` — see [Memory → PlaneFactBlock](../agents/memory.md#planefactblock-connected-central-memory).

### Durable trace buffering & retry

Trace export is **local-first, then drained to the platform**, so a platform
outage never loses spans:

1. Every span is written to local SQLite first (the durable source of truth),
   marked un-acked (`synced=0`).
2. The exporter drains un-acked spans and POSTs them to
   `/public/v1/traces/ingest`. Transient failures (connection errors, timeouts,
   HTTP 5xx) are **retried** with bounded exponential backoff (~3 attempts). A
   4xx (e.g. a bad key) is **not** retried.
3. Spans are marked acked (`synced=1`) only after a confirmed `2xx`. Anything
   still un-acked stays buffered and **re-drains on the next export** — when the
   platform comes back, the backlog flushes automatically. No reconnect hook is
   needed.

Re-sending is safe: `/traces/ingest` is **idempotent by `span_id`** (a span
already stored returns `{"ingested": 0}`), so an outage that overlaps a partial
send never double-counts.

All of this runs on a background thread — **agent execution is never blocked**
by the network.

**Bounded buffer.** The re-send queue is capped (~10,000 un-acked spans or
~7 days). Beyond that, the oldest spans are dropped from the *re-send queue* but
**kept in `local.db`** (they still appear in the Local UI); the dropped count is
logged. Local trace history is never deleted to bound the buffer.

> Upgrade note: the `synced` flag is added by an automatic, additive migration.
> Existing local traces are marked acked on upgrade, so connecting an existing
> project does **not** retroactively back-push your whole history. Use the manual
> backfill below to push historical traces on demand.

## Disconnecting

```python
fa.disconnect()
# Reverts to local-only mode
```

## Error Handling

```python
from fastaiagent._internal.errors import (
    PlatformAuthError,          # Invalid or expired API key
    PlatformConnectionError,    # Cannot reach the platform
    PlatformNotConnectedError,  # fa.connect() not called
    PlatformNotFoundError,      # Resource not found
    PlatformRateLimitError,     # Rate limit exceeded
    PlatformTierLimitError,     # Tier limit reached
)

try:
    fa.connect(api_key="bad-key")
except PlatformAuthError as e:
    print(f"Auth failed: {e}")
```

## Public API

```python
# Connection
fa.connect(api_key, target, project)
fa.disconnect()
fa.is_connected  # bool

# Prompt Registry
PromptRegistry.get(slug, version=None, source="auto")
PromptRegistry.publish(slug, content, variables=None)
PromptRegistry.refresh(slug)

# Evaluation
Dataset.from_platform(name) -> Dataset
Dataset.publish(name) -> None
EvalResults.publish(run_name=None) -> None
Scorer.from_platform(name) -> Scorer

# Replay
Replay.from_platform(trace_id) -> Replay

# Trace
TraceData.publish() -> None
```

---

## Internals

For contributors who need to understand the HTTP client, the connection lifecycle, the per-feature caching/fallback behavior, error handling, or how to add a new platform-facing endpoint, see [Platform API Internals](../internals/platform-api.md).

## Next Steps

- [Agents](../agents/index.md) — Build agents
- [Chains](../chains/index.md) — Build chains
- [Prompts](../prompts/index.md) — Manage prompts with versioning
- [Evaluation](../evaluation/index.md) — Test agent quality
- [Tracing](../tracing/index.md) — Understand the trace system
