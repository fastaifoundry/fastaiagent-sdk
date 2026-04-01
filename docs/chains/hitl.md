# Human-in-the-Loop

HITL nodes pause chain execution for human approval before proceeding. This is useful for high-stakes workflows where a human must review and approve agent output before it is sent or acted upon.

## Basic Usage

```python
from fastaiagent import Agent, Chain, LLMClient
from fastaiagent.chain import NodeType

chain = Chain("approval-pipeline")
chain.add_node("draft", agent=drafter_agent)
chain.add_node("review", type=NodeType.hitl)
chain.add_node("send", agent=sender_agent)
chain.connect("draft", "review")
chain.connect("review", "send")
```

## Custom Approval Handler

Provide a handler function that receives the node, context, and state, and returns `True` (approve) or `False` (reject):

```python
def approval_handler(node, context, state):
    draft = context["node_results"]["draft"]
    print(f"Review this draft: {draft}")
    return input("Approve? (y/n): ").lower() == "y"

result = chain.execute({"message": "Write a response"}, hitl_handler=approval_handler)
```

The handler has access to:
- `node` — the HITL node definition
- `context` — includes `node_results` from all previously executed nodes
- `state` — the current chain state

## Auto-Approve (Testing)

If no handler is provided, HITL nodes auto-approve. This is useful for testing:

```python
# No hitl_handler — auto-approves
result = chain.execute({"message": "Write a response"})
```

Or pass a lambda for quick testing:

```python
result = chain.execute(
    {"message": "Write a response"},
    hitl_handler=lambda n, c, s: True,  # Always approve
)
```

## Complete Example

A support pipeline with drafting, review, and sending:

```python
from fastaiagent import Agent, Chain, LLMClient
from fastaiagent.chain import NodeType

llm = LLMClient(provider="openai", model="gpt-4.1")

chain = Chain("support-pipeline")
chain.add_node("draft", agent=Agent(
    name="drafter", system_prompt="Draft a helpful response.", llm=llm))
chain.add_node("review", type=NodeType.hitl)
chain.add_node("send", agent=Agent(
    name="sender", system_prompt="Finalize and send the response.", llm=llm))
chain.connect("draft", "review")
chain.connect("review", "send")

def review_handler(node, context, state):
    draft = context["node_results"]["draft"]
    print(f"\n--- Draft for review ---\n{draft}\n---")
    decision = input("Approve? (y/n): ").strip().lower()
    return decision == "y"

result = chain.execute(
    {"message": "My order hasn't arrived"},
    hitl_handler=review_handler,
)
print(result.output)
```

---

## Next Steps

- [Chains](index.md) — Core chain documentation
- [Cyclic Workflows](cyclic-workflows.md) — Retry loops and exit conditions
- [Checkpointing](checkpointing.md) — Save and resume chain execution
