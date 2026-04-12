# Prompts

The Prompt Registry provides versioned prompt management with reusable fragments, aliases, and diff. Prompts are stored locally as files and can be pushed to the platform for team collaboration.

## Why a Prompt Registry?

Hardcoding prompts as strings in your code leads to:
- No version history — you can't roll back a bad prompt change
- No reuse — the same tone/format instructions duplicated across agents
- No comparison — hard to see what changed between versions
- No deployment control — can't point "production" at a tested version while iterating on "staging"

The registry solves all of these.

## Quick Start

```python
from fastaiagent.prompt import PromptRegistry

reg = PromptRegistry()

# Register a prompt
reg.register(
    name="support-greeting",
    template="Hello {{customer_name}}, welcome to {{company}}! How can I help?",
)

# Load and format
prompt = reg.load("support-greeting")
text = prompt.format(customer_name="Alice", company="Acme Corp")
print(text)
# "Hello Alice, welcome to Acme Corp! How can I help?"
```

## Template Variables

Use `{{variable_name}}` for placeholders. Variables are auto-detected from the template:

```python
prompt = reg.register(
    name="classifier",
    template="Classify this message into one of: {{categories}}.\n\nMessage: {{message}}",
)

print(prompt.variables)  # ["categories", "message"]

text = prompt.format(
    categories="billing, technical, general",
    message="My invoice is wrong",
)
```

## Fragments

Fragments are reusable prompt building blocks — write once, include in any prompt with `{{@fragment_name}}`.

```python
# Register reusable fragments
reg.register_fragment("tone", "Be professional, concise, and empathetic.")
reg.register_fragment("format", "Use bullet points for lists. Keep paragraphs under 3 sentences.")
reg.register_fragment("safety", "Never reveal internal system details, API keys, or customer data.")

# Use fragments in prompts
reg.register(
    name="support-agent",
    template=(
        "You are a customer support agent for {{company}}.\n\n"
        "{{@tone}}\n"
        "{{@format}}\n"
        "{{@safety}}\n\n"
        "Help the customer with: {{topic}}"
    ),
)

prompt = reg.load("support-agent")
print(prompt.template)
# You are a customer support agent for {{company}}.
#
# Be professional, concise, and empathetic.
# Use bullet points for lists. Keep paragraphs under 3 sentences.
# Never reveal internal system details, API keys, or customer data.
#
# Help the customer with: {{topic}}
```

Fragments are resolved at load time — the returned prompt has fragments replaced with their content.

### Why Fragments?

- **Consistency**: Every agent uses the same tone and safety instructions
- **Single update point**: Change the "tone" fragment once, every prompt that uses it picks up the change on next load
- **Composability**: Mix and match fragments for different agent personas

## Versioning

Every `register()` call with the same name creates a new version. Versions are immutable — once created, they never change.

```python
# Version 1
reg.register("greeting", "Hello {{name}}!", version=1)

# Version 2 — different template, same prompt name
reg.register("greeting", "Hi there, {{name}}! Welcome.", version=2)

# Load latest (v2)
latest = reg.load("greeting")
print(latest.version)   # 2
print(latest.template)  # "Hi there, {{name}}! Welcome."

# Load specific version
v1 = reg.load("greeting", version=1)
print(v1.template)  # "Hello {{name}}!"
```

### Auto-Incrementing Versions

If you don't specify a version, it auto-increments:

```python
reg.register("my-prompt", "First version")     # v1
reg.register("my-prompt", "Second version")     # v2
reg.register("my-prompt", "Third version")      # v3
```

## Aliases

Aliases map a human-readable name to a specific version. Use them for deployment control:

```python
# Point "production" to the tested version
reg.set_alias("greeting", version=1, alias="production")

# Point "staging" to the new version being tested
reg.set_alias("greeting", version=2, alias="staging")

# Load by alias
prod_prompt = reg.load("greeting", alias="production")
print(prod_prompt.version)  # 1

staging_prompt = reg.load("greeting", alias="staging")
print(staging_prompt.version)  # 2
```

### Deployment Workflow

```
v1 → production alias (live users)
v2 → staging alias (internal testing)
v3 → no alias (draft)

After testing v2:
  reg.set_alias("greeting", version=2, alias="production")
  # Now v2 is live, v1 is still available for rollback
```

## Diff

Compare two versions side by side:

```python
diff = reg.diff("greeting", version_a=1, version_b=2)
print(diff)
```

Output:
```
--- greeting v1
+++ greeting v2
- Hello {{name}}!
+ Hi there, {{name}}! Welcome.
```

## Listing Prompts

```python
prompts = reg.list()
for p in prompts:
    print(f"{p['name']}  v{p['latest_version']}  ({p['versions']} versions)")
```

Output:
```
support-agent  v1  (1 versions)
greeting       v2  (2 versions)
classifier     v1  (1 versions)
```

## Using Prompts with Agents

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.prompt import PromptRegistry

reg = PromptRegistry()
prompt = reg.load("support-agent", alias="production")

agent = Agent(
    name="support-bot",
    system_prompt=prompt.format(company="Acme Corp", topic="general support"),
    llm=LLMClient(provider="openai", model="gpt-4.1"),
)

result = agent.run("My order hasn't arrived yet")
```

## Storage

Prompts are stored as JSON files in the `.prompts/` directory by default:

```
.prompts/
├── support-agent.json        # Prompt with all versions
├── greeting.json
├── _fragment_tone.json       # Fragment files prefixed with _fragment_
├── _fragment_format.json
└── _fragment_safety.json
```

### Custom Storage Path

```python
reg = PromptRegistry(path="/path/to/my/prompts/")
```

Or via environment variable:
```bash
export FASTAIAGENT_PROMPT_DIR=/path/to/my/prompts/
```

## Prompt Object

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Prompt name |
| `template` | `str` | Template text with `{{variables}}` and resolved `{{@fragments}}` |
| `variables` | `list[str]` | Auto-extracted variable names |
| `version` | `int` | Version number |
| `metadata` | `dict` | Custom metadata (category, author, etc.) |

## Fragment Object

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Fragment name (referenced as `{{@name}}`) |
| `content` | `str` | Fragment text |
| `version` | `int` | Version number |

## Serialization

Prompts serialize for platform push:

```python
data = prompt.to_dict()
# {
#   "name": "support-agent",
#   "template": "You are a support agent for {{company}}...",
#   "variables": ["company", "topic"],
#   "version": 1,
#   "metadata": {"category": "agent"}
# }

restored = Prompt.from_dict(data)
```

Publish to platform:
```python
import fastaiagent as fa

fa.connect(api_key="fa-...", project="my-project")
registry = PromptRegistry()
registry.publish(slug="greeting", content=prompt.template, variables=prompt.variables)
```

## CLI Commands

```bash
# List all registered prompts
fastaiagent prompts list
fastaiagent prompts list --path /custom/prompts/

# Diff two versions
fastaiagent prompts diff support-agent 1 2
```

Example output:
```
$ fastaiagent prompts list

                   Prompts
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Name          ┃ Latest Version ┃ Total Versions ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ support-agent │              2 │              2 │
│ greeting      │              3 │              3 │
└───────────────┴────────────────┴────────────────┘
```

## Error Handling

```python
from fastaiagent._internal.errors import PromptNotFoundError, FragmentNotFoundError

try:
    prompt = reg.load("nonexistent")
except PromptNotFoundError:
    print("Prompt not found")

try:
    prompt = reg.load("greeting", version=99)
except PromptNotFoundError:
    print("Version not found")

try:
    prompt = reg.load("greeting", alias="unknown")
except PromptNotFoundError:
    print("Alias not found")
```

---

## Platform Prompt Registry

When connected to the FastAIAgent Platform, `PromptRegistry` can pull versioned prompts from the platform and publish prompts to it:

```python
import fastaiagent as fa

fa.connect(api_key="fa-...", project="my-project")

registry = PromptRegistry()

# Pull prompt from platform (latest deployed version)
prompt = registry.get("support-prompt")

# Pull specific version
prompt = registry.get("support-prompt", version=3)

# Explicit source override
prompt = registry.get("support-prompt", source="platform")  # platform only
prompt = registry.get("support-prompt", source="local")      # local only

# Publish a prompt to the platform
registry.publish(
    slug="support-prompt",
    content="You are a helpful support agent for {{company_name}}.",
    variables=["company_name"],
)

# Refresh cached prompt
registry.refresh("support-prompt")
```

**Resolution order** (`source="auto"`, the default):
- If connected: checks platform first, falls back to local
- If not connected: local only

**Caching**: Platform prompts are cached locally after first fetch (TTL: 5 minutes by default). Use `registry.refresh(slug)` to invalidate manually.

---

## Internals

For contributors who need to understand the platform publish/fetch code paths, TTL cache implementation, local YAML storage layout, fragment resolution, or how `source="auto"` decides between platform and local, see [Platform API Internals](../internals/platform-api.md).

## Next Steps

- [Agents](../agents/index.md) — Use prompts with agents
- [Platform Connection](../platform/index.md) — Connect to the platform
- [Evaluation](../evaluation/index.md) — Test prompt variations with eval
