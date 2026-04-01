# Migrating from MLflow

This guide maps MLflow concepts to FastAIAgent equivalents.

## Feature Mapping

| MLflow | FastAIAgent | Notes |
|--------|-------------|-------|
| Experiments | Evaluation runs | `evaluate()` function |
| Runs | Traces | OTel-based spans |
| Metrics | `ScorerResult` | Per-case + aggregate scoring |
| Model Registry | `PromptRegistry` | Version prompts, not models |
| Datasets | `Dataset` | JSONL/CSV based |
| MLflow Tracing | `TraceStore` | OTel-native, auto-captured |
| Evaluate | `evaluate()` | Built-in + custom scorers |

## Key Differences

1. **Agent-native**: FastAIAgent is built for AI agents, not traditional ML models.
2. **Agent Replay**: Debug agent failures with fork-and-rerun — no MLflow equivalent.
3. **Chain workflows**: Build directed graph workflows with cycles and checkpointing.
4. **Guardrails**: Built-in input/output validation, not just evaluation after the fact.

## Migration Steps

### 1. Replace MLflow Experiment Tracking

```python
# Before (MLflow)
import mlflow
with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_param("model", "gpt-4o")

# After (FastAIAgent) — evaluation is automatic
from fastaiagent.eval import evaluate
results = evaluate(
    agent_fn=my_agent.run,
    dataset="test_cases.jsonl",
    scorers=["exact_match", "contains"]
)
print(results.summary())
# exact_match: avg=0.92 pass_rate=92% (100 cases)
```

### 2. Replace MLflow Tracing

```python
# Before (MLflow)
mlflow.langchain.autolog()

# After (FastAIAgent)
import fastaiagent
fastaiagent.integrations.langchain.enable()
# Traces stored locally in SQLite, exportable via OTel
```

### 3. Replace MLflow Model Registry with Prompt Registry

```python
# Before (MLflow)
mlflow.register_model(model_uri, "my-model")

# After (FastAIAgent) — version prompts instead
from fastaiagent.prompt import PromptRegistry, Prompt
registry = PromptRegistry()
registry.save(Prompt(
    name="support-prompt",
    template="You are a support agent for {{company}}...",
    version=2,
))
```

## Next Steps

- [Evaluation Guide](../evaluation/index.md) for detailed scorer docs
- [Tracing Guide](../tracing/index.md) for OTel export
- [Prompt Registry Guide](../prompts/index.md) for versioning
