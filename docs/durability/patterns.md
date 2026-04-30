# Patterns

Production shapes that combine `interrupt()`, `@idempotent`, and the
checkpointer. Each pattern is a runnable skeleton — fill in the I/O calls
and ship.

## HITL approval flows

The shape every team adopts first.

```python
from fastaiagent import Chain, FunctionTool, SQLiteCheckpointer, idempotent, interrupt
from fastaiagent.chain.node import NodeType


@idempotent
def commit_to_ledger(amount: int, account: str) -> dict:
    # Replace with your real write. Wrapping makes resume safe.
    return {"ledger_id": f"led_{amount}_{account}", "amount": amount}


def approval_step(amount: str, account: str) -> dict:
    n = int(amount)
    if n > 50_000:
        decision = interrupt(
            reason="vp_approval",
            context={"amount": n, "account": account},
        )
        if not decision.approved:
            return {"approved": False, "reason": decision.metadata.get("reason")}
    receipt = commit_to_ledger(n, account)
    return {"approved": True, "ledger_id": receipt["ledger_id"]}


def build_chain() -> Chain:
    chain = Chain("approval-flow", checkpointer=SQLiteCheckpointer())
    chain.add_node(
        "approve",
        tool=FunctionTool(name="approval_tool", fn=approval_step),
        type=NodeType.tool,
        input_mapping={
            "amount": "{{state.amount}}",
            "account": "{{state.account}}",
        },
    )
    return chain
```

**Resume entry points:**

```bash
# Operator clicks Approve in the local UI:  Phase 10
curl -X POST http://127.0.0.1:7842/api/executions/refund-abc/resume \
    -H 'Content-Type: application/json' \
    -d '{"approved": true, "metadata": {"approver": "alice"}}'

# Or via CLI:                               Phase 9
fastaiagent resume refund-abc \
    --runner myapp.flows:build_chain \
    --value '{"approved": true, "metadata": {"approver": "alice"}}'

# Or via Python (Slack action handler, etc.):
await chain.aresume("refund-abc", resume_value=Resume(approved=True))
```

All three converge on the same atomic-claim path. Concurrent resumers
see [`AlreadyResumed`](api-reference.md#alreadyresumed); the local UI
maps that to a `409 Conflict` toast.

### Tiered approval (manager → VP)

Express each tier as its own node so each `interrupt()` claims a fresh
`pending_interrupts` row:

```python
def manager_step(amount: str) -> dict:
    n = int(amount)
    if n <= 1_000:
        return {"manager_approved": True, "amount": n}
    d = interrupt(reason="manager_approval", context={"amount": n, "tier": "manager"})
    return {"manager_approved": d.approved, "manager": d.metadata.get("approver"), "amount": n}


def vp_step(manager_approved: str, amount: str) -> dict:
    n = int(amount)
    if str(manager_approved).lower() != "true":
        return {"approved": False, "denied_by": "manager"}
    if n <= 50_000:
        return {"approved": True}
    d = interrupt(reason="vp_approval", context={"amount": n, "tier": "vp"})
    return {"approved": d.approved, "vp": d.metadata.get("approver")}


chain = Chain("tiered", checkpointer=SQLiteCheckpointer())
chain.add_node("manager", tool=FunctionTool(name="m", fn=manager_step),
               type=NodeType.tool, input_mapping={"amount": "{{state.amount}}"})
chain.add_node("vp", tool=FunctionTool(name="v", fn=vp_step), type=NodeType.tool,
               input_mapping={
                   "manager_approved": "{{state.output.manager_approved}}",
                   "amount": "{{state.amount}}",
               })
chain.connect("manager", "vp")
```

The first resume injects the manager's decision; the chain advances to
`vp`, which suspends if the amount crosses the VP threshold. A second
resume injects the VP's decision and the chain finishes. Two `interrupt()`
calls inside one node would share a single resume value on replay — keep
each tier in its own node.

## Payment processing

The canonical "must not double-charge" pattern.

```python
from fastaiagent import idempotent, interrupt


@idempotent
def charge_card(card_token: str, amount_cents: int) -> dict:
    # Stripe call — wrapped so resumes don't re-charge.
    return stripe.Charge.create(amount=amount_cents, source=card_token)


@idempotent
def send_receipt(email: str, charge_id: str) -> dict:
    return ses.send_email(...)


def payment_step(state):
    receipt = charge_card(state["card_token"], state["amount_cents"])

    if state["amount_cents"] > 100_000:  # > $1000, get human eyes
        decision = interrupt(
            reason="high_value_review",
            context={"amount_cents": state["amount_cents"], "charge_id": receipt["id"]},
        )
        if not decision.approved:
            stripe.Refund.create(charge=receipt["id"])
            return {"refunded": True, "reviewer": decision.metadata.get("approver")}

    send_receipt(state["customer_email"], receipt["id"])
    return {"charged": True, "charge_id": receipt["id"]}
```

The two `@idempotent` wrappers absorb the resume re-execution. The
`interrupt()` lets a reviewer reverse the charge before the receipt
goes out.

> **Tip:** Stripe accepts an `idempotency_key` natively. If you can pass
> one, prefer that over `@idempotent` — provider-side idempotency
> survives even a corrupted checkpoint store.

## Multi-step orchestration

A research → analysis → report pipeline where each step may pause.

```python
from fastaiagent import Chain, SQLiteCheckpointer, FunctionTool
from fastaiagent.chain.node import NodeType


def research(query: str) -> dict:
    sources = web_search(query)
    return {"sources": sources}


def analyze(sources: str) -> dict:
    decision = interrupt(
        reason="analyst_review",
        context={"source_count": len(sources)},
    )
    if not decision.approved:
        return {"halt_reason": "analyst rejected sources"}
    insights = run_analysis(sources)
    return {"insights": insights}


def report(insights: str) -> dict:
    return {"report": render_report(insights)}


chain = Chain("research-flow", checkpointer=SQLiteCheckpointer())
for name, fn in [("research", research), ("analyze", analyze), ("report", report)]:
    chain.add_node(
        name,
        tool=FunctionTool(name=f"{name}_tool", fn=fn),
        type=NodeType.tool,
    )
chain.connect("research", "analyze")
chain.connect("analyze", "report")
```

If the analyst rejects, `chain.aresume(execution_id,
resume_value=Resume(approved=False))` makes `analyze` return
`{"halt_reason": ...}` and the chain falls through `report` cleanly.

For long-running steps that may crash, use `@idempotent` per step or
split the step into a pre-write `interrupt()` and post-write
`@idempotent` call.

## Notification fan-out with retry

A node that posts to N webhooks. Retries should not re-send to the
already-delivered ones.

```python
from concurrent.futures import ThreadPoolExecutor

@idempotent(key_fn=lambda url, payload: f"webhook:{url}:{hash(str(payload))}")
def deliver(url: str, payload: dict) -> dict:
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    return {"url": url, "status": r.status_code}


def fan_out(payload, urls):
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda u: deliver(u, payload), urls))
    return {"deliveries": results}
```

Custom `key_fn` because URLs may include sensitive tokens we don't want
in the auto-generated SHA. On a retry, `deliver` returns the cached
result for already-delivered URLs and only hits the failed ones.

## Long-running batch with crash recovery

A 100-item processing chain that may take hours. We want to not start
over if the worker dies at item 73.

```python
def process_one(item_id: str, batch_id: str) -> dict:
    # ...
    return {"item_id": item_id, "ok": True}


def build_chain(items: list[str]) -> Chain:
    chain = Chain("batch", checkpointer=SQLiteCheckpointer())
    prev = None
    for item_id in items:
        node_id = f"item_{item_id}"
        chain.add_node(
            node_id,
            tool=FunctionTool(name=f"tool_{item_id}", fn=process_one),
            type=NodeType.tool,
            input_mapping={"item_id": item_id, "batch_id": "{{state.batch_id}}"},
        )
        if prev is not None:
            chain.connect(prev, node_id)
        prev = node_id
    return chain
```

If the process dies at item 73, the next run picks up at item 74 because
each completed node wrote a checkpoint. The
[crash-recovery gate](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/tests/e2e/test_gate_crash_recovery.py)
runs this exact shape 10x in CI to confirm it survives `SIGKILL`
deterministically.

## Anti-patterns

| Don't | Why |
|---|---|
| Call `interrupt()` more than once in the same node | A single `Resume` value is in scope on replay; the second `interrupt()` would also receive it instead of suspending. Split each gate into its own node — see [tiered approval](#tiered-approval-manager-vp). |
| Read the current time inside a node and branch on it | Time changes between pause and resume; capture at first execution. See [side effects](side-effects.md#time-dependent-decisions). |
| Call `interrupt()` from inside an `@idempotent` function | The cache hit on resume short-circuits the function and `interrupt()` is never re-evaluated. Move `interrupt()` outside. |
| Suspend across distinct logical batches in one execution_id | Each batch should be its own execution. Sharing one ID makes `chain.aresume(...)` ambiguous about *which* pause to claim. |
| Store secrets in `interrupt(context=…)` | Context is persisted in the database and visible in the `/approvals` UI. Pass a reference (a token-id, not the token). |
| Skip `setup()` and assume the DB exists | Cheap to call; saves you from a 3am paging when a fresh deploy hits an empty DB. |

## See also

- [Side effects & idempotency](side-effects.md) — the deeper dive on
  why these wrappers exist.
- [Multi-agent durability](multi-agent.md) — same patterns inside
  Agent / Swarm / Supervisor.
- [API reference](api-reference.md) — exact signatures for every
  primitive shown here.
