# Cyclic Workflows

Chains support cyclic edges (retry loops) that repeat until a condition is met or a maximum iteration count is reached. This is useful for quality-checking agent output and retrying if it doesn't meet a threshold.

## Creating a Retry Loop

```python
from fastaiagent import Agent, Chain, LLMClient

chain = Chain("research-with-retry")
chain.add_node("research", agent=research_agent)
chain.add_node("evaluate", agent=evaluator_agent)
chain.add_node("respond", agent=responder_agent)

chain.connect("research", "evaluate")
chain.connect(
    "evaluate", "research",
    max_iterations=3,                    # Max 3 retries
    exit_condition="quality >= 0.8",     # Exit loop when quality is high enough
)
chain.connect("evaluate", "respond", condition="quality >= 0.8")
```

In this example:
1. The `research` agent produces output
2. The `evaluate` agent scores the quality
3. If `quality < 0.8`, the loop goes back to `research` (up to 3 times)
4. If `quality >= 0.8`, execution continues to `respond`

## Cycle Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_iterations` | Upper bound on loop count | Required for cyclic edges |
| `exit_condition` | Expression to exit early | None (runs until max) |

When `max_iterations` is exceeded, a `ChainCycleError` is raised.

## Exit Conditions

Exit condition expressions support the same operators as conditional edges:

- `==`, `!=`, `>`, `<`, `>=`, `<=`
- `contains`, `startswith`

Values are resolved from the chain state. For example, `quality >= 0.8` checks the `quality` key in the current chain state.

## Chain Validation

The chain validator checks that all cyclic edges have `max_iterations` set:

```python
errors = chain.validate()
# Reports: "Cyclic edge evaluate→research missing max_iterations"
```

## Error Handling

```python
from fastaiagent._internal.errors import ChainCycleError

try:
    result = chain.execute({"input": "data"})
except ChainCycleError as e:
    print(f"Cycle limit hit: {e}")
    # "Cycle exceeded max_iterations (3) on edge evaluate→research"
```

---

## Next Steps

- [Chains](index.md) — Core chain documentation
- [Checkpointing](checkpointing.md) — Save and resume chain execution
- [Human-in-the-Loop](hitl.md) — Pause chains for human approval
