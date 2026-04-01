# Trajectory Scoring

Evaluate the **path** an agent took — which tools it called and in what order. Trajectory scorers help you ensure agents are not only producing correct output but also following the right process.

## ToolUsageAccuracy

Did the agent use the correct tools?

```python
from fastaiagent.eval.trajectory import ToolUsageAccuracy

scorer = ToolUsageAccuracy()
result = scorer.score(
    input="", output="",
    actual_trajectory=["search", "calculate"],
    expected_trajectory=["search", "calculate", "format"],
)
# score = 2/3 = 0.667 (used 2 of 3 expected tools)
```

## StepEfficiency

Did the agent solve the problem in the expected number of steps?

```python
from fastaiagent.eval.trajectory import StepEfficiency

scorer = StepEfficiency()
result = scorer.score(
    input="", output="",
    actual_steps=6,
    expected_steps=3,
)
# score = 3/6 = 0.5 (took twice as many steps as expected)
```

## PathCorrectness

Did the agent follow the correct sequence? Uses Longest Common Subsequence to measure ordering fidelity.

```python
from fastaiagent.eval.trajectory import PathCorrectness

scorer = PathCorrectness()
result = scorer.score(
    input="", output="",
    actual_trajectory=["search", "validate", "calculate", "respond"],
    expected_trajectory=["search", "calculate", "respond"],
)
# LCS = ["search", "calculate", "respond"] → score = 3/3 = 1.0
```

## CycleEfficiency

Did the agent avoid unnecessary repeated tool calls?

```python
from fastaiagent.eval.trajectory import CycleEfficiency

scorer = CycleEfficiency()
result = scorer.score(
    input="", output="",
    actual_trajectory=["search", "search", "search", "respond"],
)
# 2 repeated consecutive calls out of 4 → score = 1.0 - 2/4 = 0.5
```

## Using in Evaluation

Pass trajectory scorers to `evaluate()` like any other scorer. Your dataset items should include `expected_trajectory` fields:

```python
from fastaiagent.eval import evaluate
from fastaiagent.eval.trajectory import ToolUsageAccuracy, PathCorrectness

results = evaluate(
    agent_fn=my_agent.run,
    dataset=[
        {
            "input": "Calculate 15% of 200",
            "expected": "30",
            "expected_trajectory": ["calculate"],
        },
    ],
    scorers=[ToolUsageAccuracy(), PathCorrectness()],
)
```

---

## Next Steps

- [Evaluation](index.md) — Core evaluation documentation
- [LLM Judge](llm-judge.md) — Use an LLM to evaluate output quality
- [Session Scoring](session-scoring.md) — Evaluate multi-turn conversations
