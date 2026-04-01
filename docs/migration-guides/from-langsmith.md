# Migrating from LangSmith

This guide maps LangSmith concepts to FastAIAgent equivalents and shows how to migrate.

## Feature Mapping

| LangSmith | FastAIAgent | Notes |
|-----------|-------------|-------|
| Traces | `TraceStore` / OTel spans | Local-first, OTel-native |
| Runs | Spans within a trace | Same concept, different name |
| Datasets | `Dataset` | JSONL/CSV based |
| Evaluators | `Scorer` | Built-in + custom + LLM judge |
| Hub (prompts) | `PromptRegistry` | Local + platform sync |
| Feedback | Eval results | Programmatic, not manual |
| Playground | Agent Replay | Fork-and-rerun (more powerful) |

## Key Differences

1. **Local-first**: FastAIAgent stores traces locally in SQLite. No mandatory cloud dependency.
2. **OTel-native**: Export traces to Datadog, Grafana, Jaeger — not locked into one vendor.
3. **Agent Replay**: Step through any trace, fork at any point, modify state, and rerun. LangSmith has no equivalent.
4. **Build + Observe**: FastAIAgent is both an agent framework AND an observability tool.

## Migration Steps

### 1. Replace LangSmith Tracing

```python
# Before (LangSmith)
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls-..."

# After (FastAIAgent)
import fastaiagent
fastaiagent.integrations.langchain.enable()
# Traces stored locally — no API key needed
```

### 2. Replace LangSmith Datasets

```python
# Before (LangSmith)
from langsmith import Client
client = Client()
dataset = client.create_dataset("my-eval")

# After (FastAIAgent)
from fastaiagent.eval import Dataset
dataset = Dataset.from_jsonl("test_cases.jsonl")
# Or inline:
dataset = Dataset.from_list([
    {"input": "Hello", "expected_output": "Hi there!"},
])
```

### 3. Replace LangSmith Evaluators

```python
# Before (LangSmith)
from langsmith.evaluation import evaluate
evaluate(my_chain, data="my-eval", evaluators=["qa"])

# After (FastAIAgent)
from fastaiagent.eval import evaluate
results = evaluate(
    agent_fn=my_agent.run,
    dataset="test_cases.jsonl",
    scorers=["exact_match", "contains"]
)
print(results.summary())
```

### 4. Replace LangSmith Hub (Prompts)

```python
# Before (LangSmith)
from langchain import hub
prompt = hub.pull("my-prompt")

# After (FastAIAgent)
from fastaiagent.prompt import PromptRegistry
registry = PromptRegistry()
prompt = registry.get("my-prompt")
# With versioning:
prompt_v2 = registry.get("my-prompt", version=2)
```

## Next Steps

- [Tracing Guide](../tracing/index.md)
- [Evaluation Guide](../evaluation/index.md)
- [Prompt Registry Guide](../prompts/index.md)
