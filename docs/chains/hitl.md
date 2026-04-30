# Human-in-the-Loop

There are two ways to gate a chain on a human decision.

| Mode | When to use | API |
|---|---|---|
| **Suspending HITL** (`interrupt()`) | Approvals that may take seconds, hours, or days. Process exits cleanly between pause and resume. | `interrupt(reason, context)` inside any node + `chain.resume(execution_id, resume_value=Resume(approved=True))` |
| **Blocking HITL** (`NodeType.hitl` + handler) | Inline CLI-style prompts that block in the calling process. Useful for tests and local scripts. | `chain.add_node("review", type=NodeType.hitl)` + `chain.execute(..., hitl_handler=fn)` |

The two modes are independent — pick the one that matches the workflow.

> **For end-to-end durability** (suspending HITL plus crash recovery,
> the `/approvals` UI, Postgres in production, the `@idempotent`
> decorator that makes resumed nodes safe to re-run, and resume from
> Python / HTTP / CLI), see the dedicated
> [Durability section](../durability/index.md).

## Suspending HITL — `interrupt()`

Calling `interrupt(reason, context)` inside any node suspends the chain
cleanly. The executor persists an *interrupted* checkpoint plus a row in
`pending_interrupts`, and `chain.execute(...)` returns a `ChainResult` with
`status="paused"`. The Python process can exit. A separate
`chain.resume(...)` call (in a different process, hours later) injects a
`Resume` value and re-runs the suspended node — this time `interrupt()`
returns the value instead of raising.

```python
from fastaiagent import Chain, FunctionTool, Resume, interrupt
from fastaiagent.chain.node import NodeType

def approval(amount: int):
    if amount > 10_000:
        decision = interrupt(
            reason="manager_approval",
            context={"amount": amount, "policy": "high-value"},
        )
        return {"approved": decision.approved,
                "approver": decision.metadata.get("approver")}
    return {"approved": True, "auto": True}

chain = Chain("payments")
chain.add_node("approval",
               tool=FunctionTool(name="approval_tool", fn=approval),
               type=NodeType.tool,
               input_mapping={"amount": "{{state.amount}}"})
# ... wire up other nodes ...

# First run — suspends.
result = chain.execute({"amount": 50_000}, execution_id="payment-abc")
assert result.status == "paused"

# Sometime later (different process is fine):
result = await chain.resume(
    "payment-abc",
    resume_value=Resume(approved=True, metadata={"approver": "alice"}),
)
assert result.status == "completed"
```

### Frozen-context invariant

The `context` dict you pass to `interrupt()` is **JSON-serialized at suspend
time** and stored verbatim in the checkpoint and `pending_interrupts` row.
On resume, the executor does **not** recompute it — the human approved a
specific snapshot, and that snapshot is what the resumer sees.

This matters because the world may have changed between pause and resume:
balances move, prices update, customers cancel. If your node needs the
*current* values when it resumes, read them from chain `state` (which is
also rehydrated on resume) instead of relying on the frozen `context`.

### Three resume entry points

`Chain.resume(...)` is the Python API. The same internal path is reachable
from two more places:

```bash
# CLI — useful for ad-hoc operator runs and ops scripts.
fastaiagent resume <execution-id> \
    --runner myapp.chains:my_chain \
    --value '{"approved": true, "metadata": {"approver": "alice"}}'

fastaiagent list-pending           # rich table of every paused workflow
fastaiagent inspect <execution-id> # checkpoint history for one execution
```

```python
# HTTP — the /approvals UI (Phase 10) and any external system call this.
from fastaiagent.ui.server import build_app

# Pass the chains/agents the server needs to resume.
app = build_app(db_path=..., runners=[my_chain, my_agent, my_swarm])
```

```bash
# POST /api/executions/{execution_id}/resume
curl -X POST http://127.0.0.1:7842/api/executions/refund-abc/resume \
    -H 'Content-Type: application/json' \
    -d '{"approved": true, "metadata": {"approver": "alice"}}'
```

All three converge on the same atomic-claim semantics: concurrent
resumers see `AlreadyResumed` (CLI exit code 2; HTTP `409 Conflict`).

### Side effects before `interrupt()`

Because resume re-runs the suspended node from the top, anything you do
*before* the `interrupt()` call runs twice. If your node charges a card,
sends an email, or makes any other side-effectful call before suspending,
wrap it with [`@idempotent`](idempotency.md) — that's exactly what the
decorator exists for.

### Atomic resume claim

`chain.resume(execution_id, resume_value=…)` atomically deletes the
`pending_interrupts` row before re-running the suspended node. If the row
is already gone (another resumer beat us, or the chain was never
suspended), it raises `AlreadyResumed`. This is the safety net against
double-clicking an "Approve" button.

```python
from fastaiagent import AlreadyResumed

try:
    await chain.resume(execution_id, resume_value=Resume(approved=True))
except AlreadyResumed:
    print("This approval was already processed.")
```

## Blocking HITL — inline handler

Use this for CLI tooling, tests, or scripts where the chain runs to
completion in a single process and a human is sitting at the terminal.

### Basic Usage

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

### Custom Approval Handler

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

### Rejection Behavior

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

### Auto-Approve (Testing)

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

### Complete Example

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
