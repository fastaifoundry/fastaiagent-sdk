# Session Scoring

Evaluate multi-turn conversations as a whole. Session scorers assess coherence across turns and whether the conversation achieved its goal.

## ConversationCoherence

Are the agent's responses coherent across turns?

```python
from fastaiagent.eval.session import ConversationCoherence

scorer = ConversationCoherence()
result = scorer.score(
    input="", output="final response",
    turns=[
        {"role": "user", "content": "What is X?"},
        {"role": "assistant", "content": "X is..."},
        {"role": "user", "content": "Tell me more"},
        {"role": "assistant", "content": "Additionally..."},
    ],
)
```

The scorer analyzes the conversation flow to ensure the assistant's responses are contextually consistent and build upon previous turns.

## GoalCompletion

Did the conversation achieve its goal?

```python
from fastaiagent.eval.session import GoalCompletion

scorer = GoalCompletion()
result = scorer.score(
    input="", output="Your order ships tomorrow via FedEx.",
    goal="Provide shipping information for the customer's order",
)
# Measures keyword overlap between output and goal
```

## Using in Evaluation

Pass session scorers to `evaluate()`. Your dataset items should include `conversation` or `goal` fields as appropriate:

```python
from fastaiagent.eval import evaluate
from fastaiagent.eval.session import ConversationCoherence, GoalCompletion

results = evaluate(
    agent_fn=my_agent.run,
    dataset=[
        {
            "input": "Where is my order?",
            "goal": "Provide shipping information",
            "conversation": [
                {"role": "user", "content": "Where is my order?"},
                {"role": "assistant", "content": "Let me look that up..."},
            ],
        },
    ],
    scorers=[GoalCompletion()],
)
```

---

## Next Steps

- [Evaluation](index.md) — Core evaluation documentation
- [LLM Judge](llm-judge.md) — Use an LLM to evaluate output quality
- [Trajectory Scoring](trajectory-scoring.md) — Evaluate the path an agent took
