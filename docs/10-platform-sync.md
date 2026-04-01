# Platform Sync

The SDK can push agents, chains, tools, guardrails, and prompts to the FastAIAgent Platform for visual editing, production monitoring, and team collaboration. Sync is **unidirectional** — SDK to platform only.

## How It Works

```
SDK (your code)                         Platform (remote server)
─────────────────                       ─────────────────────────
Agent.to_dict()  ──→  POST /public/v1/sdk/push  ──→  Database
Chain.to_dict()       (X-API-Key auth)               Visual Editor
Tool.to_dict()        (rate limited)                  Monitoring
Guardrail.to_dict()   (domain scoped)                 Collaboration
Prompt.to_dict()
```

The SDK serializes resources to canonical JSON format and sends them to the platform's public API. The platform creates or updates resources via **upsert-by-name** — if a resource with the same name exists in the project, it's updated; otherwise it's created.

## Setup

### 1. Create an Account

**SaaS (hosted):**
- Go to [https://app.fastaiagent.net](https://app.fastaiagent.net)
- Sign up with email or SSO

**On-premise (self-hosted):**
- Navigate to your organization's FastAIAgent instance URL
- Sign up or log in with your corporate credentials

### 2. Create a Domain

A domain is the top-level workspace that isolates resources and team access.

- Go to Settings → Domains → Create Domain
- Give it a name (e.g., "Engineering", "Customer Support")

### 3. Create a Project

A project groups related agents, chains, and tools within a domain.

- Navigate to your domain
- Go to Projects → Create Project
- Give it a name (e.g., "Support Bots", "Internal Tools")

### 4. Create an API Key

- Go to Settings → API Keys → Create Key
- Select the **domain** and optionally a specific **project**
- Select scopes: `agent:write`, `chain:write`, `tool:write`, `guardrail:write`, `prompt:write`
- Copy the key (shown only once): `fa_k_...`

> **Important:** The key is shown only at creation time. Store it securely. If lost, create a new key.

The API key determines which domain and project your pushes go to. Resources are created in the key's project.

### 5. Connect

```python
from fastaiagent import FastAI

fa = FastAI(
    api_key="fa_k_...",
    target="https://app.fastaiagent.net",  # Or your self-hosted URL
)
```

**Environment variables** (alternative to constructor args):
```bash
export FASTAIAGENT_API_KEY=fa_k_...
export FASTAIAGENT_TARGET=https://app.fastaiagent.net
```

### FastAI Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | Required | Platform API key with write scopes |
| `target` | `str` | `https://app.fastaiagent.net` | Platform URL |
| `project` | `str \| None` | None | Project name (uses API key's default project) |
| `timeout` | `int` | 30 | HTTP timeout in seconds |

## Pushing Resources

### Push an Agent

Automatically pushes the agent's tools and guardrails as dependencies:

```python
from fastaiagent import Agent, FunctionTool, LLMClient, FastAI
from fastaiagent.guardrail import no_pii

def search(query: str) -> str:
    """Search the knowledge base."""
    return f"Results for: {query}"

agent = Agent(
    name="support-bot",
    system_prompt="You are a helpful support agent. Use tools to find answers.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    tools=[FunctionTool(name="search", fn=search)],
    guardrails=[no_pii()],
)

fa = FastAI(api_key="fa_k_...")
result = fa.push(agent)

print(result.name)                  # "support-bot"
print(result.resource_type)         # "agent"
print(result.created)               # True (first push) or False (update)
print(result.dependencies_pushed)   # ["tool:search", "guardrail:no_pii"]
```

**What gets pushed:**
- The agent (name, system_prompt, config)
- All tools (name, description, tool_type, parameters, config)
- All guardrails (name, type, config, position, blocking mode)

### Push a Chain

Automatically pushes node agents and their tools/guardrails:

```python
from fastaiagent import Agent, Chain, LLMClient, FastAI

chain = Chain("support-pipeline")
chain.add_node("research", agent=Agent(
    name="researcher", system_prompt="Research the topic.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
))
chain.add_node("respond", agent=Agent(
    name="responder", system_prompt="Write a response.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
))
chain.connect("research", "respond")

fa = FastAI(api_key="fa_k_...")
result = fa.push(chain)

print(result.name)                  # "support-pipeline"
print(result.dependencies_pushed)   # ["agent:researcher", "agent:responder"]
```

The chain appears in the platform's **visual editor** with nodes and edges rendered.

### Push a Tool

```python
from fastaiagent import FunctionTool, FastAI

tool = FunctionTool(
    name="calculate",
    description="Evaluate a math expression",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
)

fa = FastAI(api_key="fa_k_...")
result = fa.push(tool)
```

### Push a Guardrail

```python
from fastaiagent.guardrail import Guardrail, GuardrailType
from fastaiagent import FastAI

guardrail = Guardrail(
    name="no_urls",
    guardrail_type=GuardrailType.regex,
    config={"pattern": r"https?://", "should_match": False},
    description="Blocks URLs in output",
)

fa = FastAI(api_key="fa_k_...")
result = fa.push(guardrail)
```

### Push a Prompt

Creates the prompt in the platform's prompt registry with a version:

```python
from fastaiagent.prompt import Prompt
from fastaiagent import FastAI

prompt = Prompt(
    name="support-greeting",
    template="Hello {{customer_name}}, welcome to {{company}}!",
    metadata={"category": "agent", "description": "Customer greeting"},
)

fa = FastAI(api_key="fa_k_...")
result = fa.push(prompt)
```

## Batch Push

Push multiple resources in a single request:

```python
results = fa.push_all([agent, chain, tool, guardrail, prompt])

for r in results:
    print(f"{r.resource_type}:{r.name} — {'created' if r.created else 'updated'}")
```

Dependencies are resolved automatically — tools are pushed before agents, agents before chains.

## Upsert Behavior

Push uses **upsert-by-name** semantics:

| Scenario | Behavior |
|----------|----------|
| Name doesn't exist in project | **Creates** new resource |
| Name already exists in project | **Updates** existing resource |
| Agent has new tools | Creates new tools, updates agent's tool list |
| Push same agent twice | Updates the agent, increments version |

```python
# First push — creates
result = fa.push(agent)
print(result.created)  # True

# Second push — updates
agent.system_prompt = "Updated prompt"
result = fa.push(agent)
print(result.created)  # False (updated)
```

## PushResult

| Field | Type | Description |
|-------|------|-------------|
| `resource_type` | `str` | `"agent"`, `"chain"`, `"tool"`, `"guardrail"`, `"prompt"` |
| `name` | `str` | Resource name |
| `platform_id` | `str` | Platform-assigned UUID |
| `created` | `bool` | `True` if new, `False` if updated |
| `dependencies_pushed` | `list[str]` | Auto-pushed dependencies (e.g., `["tool:search"]`) |

## API Key Scopes

The API key must have the appropriate write scopes:

| Scope | Allows |
|-------|--------|
| `agent:write` | Push agents |
| `chain:write` | Push chains |
| `tool:write` | Push tools |
| `guardrail:write` | Push guardrails |
| `prompt:write` | Push prompts |

Pushing an agent also requires `tool:write` and `guardrail:write` for its dependencies. Pushing a chain also requires `agent:write`.

**Recommended:** Create a single SDK key with all write scopes.

## Authentication

The SDK authenticates via the `X-API-Key` HTTP header. The platform validates:

1. Key exists and is active
2. Key is not expired
3. Key has the required scope
4. Resource belongs to the key's domain

## Rate Limiting

The platform enforces per-key rate limits:

| Limit | Default |
|-------|---------|
| Requests per minute (RPM) | 60 |
| Requests per day (RPD) | 10,000 |

When rate limited, the SDK raises `PlatformRateLimitError` with the `Retry-After` duration.

## Offline Cache

When the platform is unreachable, pushes can be buffered locally and retried later:

```python
from fastaiagent._platform.cache import OfflineCache

cache = OfflineCache()

# Buffer a push for later
cache.buffer_push("agent", agent.to_dict())

# Check buffered items
pending = cache.get_buffered_pushes()
print(f"{len(pending)} pushes waiting")

# Clear after successful manual retry
cache.clear_buffer()
```

## CLI

```bash
# Push via CLI (module:object syntax planned)
fastaiagent push --api-key fa_k_... --target https://app.fastaiagent.net --agent myapp:agent

# Using environment variables
export FASTAIAGENT_API_KEY=fa_k_...
export FASTAIAGENT_TARGET=https://app.fastaiagent.net
fastaiagent push --agent myapp:support_agent
```

## Error Handling

```python
from fastaiagent._internal.errors import (
    PlatformAuthError,        # Invalid or expired API key
    PlatformConnectionError,  # Cannot reach the platform
    PlatformNotFoundError,    # Resource not found
    PlatformRateLimitError,   # Rate limit exceeded
    PlatformTierLimitError,   # Tier limit reached
)

try:
    result = fa.push(agent)
except PlatformAuthError as e:
    print(f"Auth failed: {e}")
    # "Invalid API key" or "Insufficient permissions: API key lacks required scope: agent:write"
except PlatformConnectionError as e:
    print(f"Cannot connect: {e}")
    # "Cannot connect to platform. Check your internet connection..."
except PlatformRateLimitError as e:
    print(f"Rate limited: {e}")
    # "Rate limit exceeded. Retry after 30 seconds."
except PlatformTierLimitError as e:
    print(f"Tier limit: {e}")
    # "Tier limit reached. Upgrade at https://app.fastaiagent.net/billing"
```

## What Appears on the Platform

After pushing:

| Resource | Platform View |
|----------|--------------|
| Agent | Listed in Agents page, editable, executable |
| Chain | Visual editor with nodes and edges rendered |
| Tool | Listed in Tools page, testable |
| Guardrail | Listed in Guardrails page, attachable to agents/chains |
| Prompt | Listed in Prompt Registry with version history |

## Complete Example

```python
from fastaiagent import Agent, Chain, FunctionTool, LLMClient, FastAI
from fastaiagent.prompt import Prompt
from fastaiagent.guardrail import no_pii

# 1. Define resources
def lookup(query: str) -> str:
    return f"Info about: {query}"

agent = Agent(
    name="demo-agent",
    system_prompt="Use tools to answer questions.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    tools=[FunctionTool(name="lookup", fn=lookup)],
    guardrails=[no_pii()],
)

chain = Chain("demo-pipeline")
chain.add_node("process", agent=agent)

prompt = Prompt(
    name="demo-prompt",
    template="Hello {{name}}, how can I help?",
    metadata={"category": "agent"},
)

# 2. Connect and push everything
fa = FastAI(api_key="fa_k_...", target="http://localhost:8001")

results = fa.push_all([agent, chain, prompt])
for r in results:
    print(f"{'Created' if r.created else 'Updated'}: {r.resource_type}:{r.name}")

# Output:
# Created: agent:demo-agent
# Created: chain:demo-pipeline
# Created: prompt:demo-prompt
```
