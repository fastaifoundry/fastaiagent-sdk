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

## Rejection Behavior

When the handler returns `False`, the HITL node records `approved=False` on its result dict, but **the chain continues running**. Rejection does not halt execution. This is by design — the HITL node captures the decision, but downstream nodes decide what to do with it.

If you need halt-on-reject, combine a HITL node with a **condition node** that branches on the `approved` field:

```python
chain.add_node("draft", agent=drafter_agent)
chain.add_node("review", type=NodeType.hitl)
chain.add_node("check_approval", type=NodeType.condition,
               conditions=[{"expression": "{{state.output.approved}} == True", "handle": "approved"}])
chain.add_node("send", agent=sender_agent)
chain.add_node("abort", type=NodeType.end)

chain.connect("draft", "review")
chain.connect("review", "check_approval")
chain.connect("check_approval", "send", condition="approved")
chain.connect("check_approval", "abort")  # Default route if not approved
```

You can inspect the approval decision after execution:

```python
result = chain.execute({"message": "..."}, hitl_handler=my_handler)
review_result = result.node_results.get("review", {})
print(review_result.get("approved"))   # True or False
print(review_result.get("message"))    # "Auto-approved (no HITL handler)" if no handler
```

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
