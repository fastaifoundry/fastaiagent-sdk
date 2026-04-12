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

## What Does NOT Flow Through `fa.connect()`

| Capability | Reason |
|-----------|--------|
| Agent definitions | Agents are code, not config to push |
| Tool implementations | Tools are Python functions in SDK |
| Guardrail definitions | Guardrails are code-configured in SDK |
| Chain definitions | Chains are code in SDK |
| KB data/documents | LocalKB is local |
| LLM endpoint credentials | SDK manages its own API keys |

## Offline / Disconnected Behavior

Every service degrades gracefully when the platform is unreachable:

| Service | Connected | Disconnected |
|---------|-----------|-------------|
| Traces | Send to platform + local SQLite | Local SQLite only |
| Prompts | Fetch from platform (cached locally) | Use local cache or local files |
| Eval datasets | Pull from platform | Use local JSONL/CSV |
| Eval results | Publish to platform | Store locally, publish later |
| Replay | Pull platform traces | Local traces only |

No operation fails because the platform is down.

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
