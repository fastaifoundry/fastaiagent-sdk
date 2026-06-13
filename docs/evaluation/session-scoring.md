# Session Scoring

Evaluate multi-turn conversations as a whole. Session scorers assess coherence across turns and whether the conversation achieved its goal. Coherence and goal scoring run as fast heuristics by default, or as LLM judges with `mode="llm"`; three more metrics — knowledge retention, role adherence, and conversation relevancy — are LLM-judged.

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

## LLM-judged mode

`ConversationCoherence` and `GoalCompletion` default to fast, zero-dependency heuristics (`mode="heuristic"`). Pass `mode="llm"` to judge with an LLM instead — useful when nuance matters more than speed. The heuristic default is unchanged, and `threshold` governs pass/fail on the LLM's 0–1 score (default `0.5`).

```python
from fastaiagent.eval import ConversationCoherence, GoalCompletion
from fastaiagent import LLMClient

llm = LLMClient(provider="openai", model="gpt-4o-mini")

coherence = ConversationCoherence(mode="llm", llm=llm).score(input="", output="", turns=turns)
goal = GoalCompletion(mode="llm", llm=llm).score(
    input="", output="", goal="Provide the carrier and arrival day", turns=turns
)
```

## LLM-judged turn metrics

Three additional metrics judge specific conversational qualities with an LLM. Each takes `turns` and is always LLM-judged (`threshold` default `0.7`).

```python
from fastaiagent.eval import KnowledgeRetention, RoleAdherence, ConversationRelevancy

# Does the agent reuse info the user gave earlier (no re-asking / contradiction)?
KnowledgeRetention(llm=llm).score(input="", output="", turns=turns)

# Does the agent stay in its assigned role? (role via constructor or a `role` kwarg)
RoleAdherence(role="a formal banking assistant", llm=llm).score(input="", output="", turns=turns)

# Are the agent's replies relevant to each user turn?
ConversationRelevancy(llm=llm).score(input="", output="", turns=turns)
```

`RoleAdherence` returns `score=0.0` with reason `"No role specified"` when no role is given.

## Using these scorers

Session scorers operate on conversation data passed through keyword arguments —
`turns` for `ConversationCoherence`, `goal` for `GoalCompletion`. Because
`evaluate()`'s dataset loop only forwards `input`/`expected` per case (not
`conversation`/`goal`), call `.score(...)` directly with the conversation you
captured:

```python
from fastaiagent.eval import ConversationCoherence, GoalCompletion

turns = [
    {"role": "user", "content": "Where is my order?"},
    {"role": "assistant", "content": "Your order ships tomorrow via FedEx."},
]

coherence = ConversationCoherence().score(input="", output="", turns=turns)
goal = GoalCompletion().score(
    input="",
    output=turns[-1]["content"],
    goal="Provide shipping information for the customer's order",
)
print("coherence:", coherence.score, "| goal:", goal.score)
```

See `examples/77_session_eval.py` (heuristic) and `examples/82_llm_session_metrics.py`
(LLM-judged mode + the new turn metrics) for runnable end-to-end scripts.

---

## Next Steps

- [Evaluation](index.md) — Core evaluation documentation
- [LLM Judge](llm-judge.md) — Use an LLM to evaluate output quality
- [Trajectory Scoring](trajectory-scoring.md) — Evaluate the path an agent took
