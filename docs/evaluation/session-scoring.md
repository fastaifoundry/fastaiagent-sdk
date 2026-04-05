# Session Scoring

Evaluate multi-turn conversations as a whole. Session scorers assess coherence across turns and whether the conversation achieved its goal.

## ConversationCoherence

Are the agent's responses coherent across turns? Detects self-contradictions and topic drift by analyzing consecutive turns.

Checks for:
- **Self-contradiction signals** — phrases like "actually, I was wrong", "let me correct", etc.
- **Topic drift** — low vocabulary overlap between consecutive turns

```python
from fastaiagent.eval.session import ConversationCoherence

scorer = ConversationCoherence()

# Coherent conversation
result = scorer.score(
    input="", output="final response",
    turns=[
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Who created it?"},
        {"role": "assistant", "content": "Python was created by Guido van Rossum."},
    ],
)
# score ≈ 1.0 (no contradictions, on-topic)

# Contradictory conversation
result = scorer.score(
    input="", output="final response",
    turns=[
        {"content": "The capital of France is London."},
        {"content": "Actually, I was wrong. The capital is Paris."},
    ],
)
# score ≈ 0.5 (contradiction detected)
```

## GoalCompletion

Did the conversation achieve its goal? Uses keyword recall (with stop-word filtering), key-phrase matching, and checklist detection for structured goals.

```python
from fastaiagent.eval.session import GoalCompletion

scorer = GoalCompletion()

# Simple goal
result = scorer.score(
    input="", output="Your order ships tomorrow via FedEx.",
    goal="Provide shipping information for the customer's order",
)

# Structured checklist goal
result = scorer.score(
    input="",
    output="Install Python 3.12, create a venv, and run pip install.",
    goal="1. Install Python\n2. Set up virtual environment\n3. Install dependencies",
)
# Detects checklist items and scores each separately
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
