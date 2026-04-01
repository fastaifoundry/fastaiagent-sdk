# Migrating from Langfuse

This guide maps Langfuse concepts to FastAIAgent equivalents.

## Feature Mapping

| Langfuse | FastAIAgent | Notes |
|----------|-------------|-------|
| Traces | `TraceStore` / OTel spans | Local-first, OTel-native |
| Generations | LLM spans | Auto-captured |
| Spans | OTel spans | Standard format |
| Scores | `Scorer` + `EvalResults` | Programmatic scoring |
| Prompts | `PromptRegistry` | Local + versioning + fragments |
| Datasets | `Dataset` | JSONL/CSV |
| Dashboard | FastAIAgent Platform | Optional cloud UI |

## Key Differences

1. **Agent framework included**: FastAIAgent builds agents, not just observes them.
2. **Agent Replay**: Fork-and-rerun debugging — Langfuse has no equivalent.
3. **OTel-native**: Export to any OTel-compatible backend, not just Langfuse cloud.
4. **Local-first**: Works fully offline, no account required.

## Migration Steps

### 1. Replace Langfuse Tracing

```python
# Before (Langfuse)
from langfuse import Langfuse
langfuse = Langfuse(public_key="pk-...", secret_key="sk-...")
trace = langfuse.trace(name="my-agent")

# After (FastAIAgent) — automatic tracing
from fastaiagent import Agent, LLMClient
agent = Agent(
    name="my-agent",
    llm=LLMClient(provider="openai", model="gpt-4o"),
)
result = agent.run("Hello", trace=True)
# Traces stored locally automatically
```

### 2. Replace Langfuse Prompt Management

```python
# Before (Langfuse)
prompt = langfuse.get_prompt("my-prompt")
compiled = prompt.compile(variable="value")

# After (FastAIAgent)
from fastaiagent.prompt import PromptRegistry
registry = PromptRegistry()
prompt = registry.get("my-prompt")
rendered = prompt.render(variable="value")
```

### 3. Replace Langfuse Scores

```python
# Before (Langfuse)
langfuse.score(trace_id="...", name="quality", value=0.9)

# After (FastAIAgent) — programmatic evaluation
from fastaiagent.eval import evaluate
results = evaluate(
    agent_fn=my_agent.run,
    dataset="test_cases.jsonl",
    scorers=["exact_match"]
)
```

## Next Steps

- [Tracing Guide](../tracing/index.md)
- [Prompt Registry Guide](../prompts/index.md)
- [Agent Replay Guide](../replay/index.md)
