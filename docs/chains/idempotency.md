# Idempotency — `@idempotent`

When a node calls `interrupt()`, the chain suspends at that node. On resume
the executor re-runs the **same node from the top** so `interrupt()` can
return the resume value instead of raising. Anything that happened before
that `interrupt()` call — charging a card, sending an email, posting to
Slack — runs a second time.

The `@idempotent` decorator caches a function's result for the lifetime of
an execution, so a side effect runs once even if the node re-runs.

## The footgun

```python
def approval_node(state):
    receipt = charge_customer(state["amount"])  # side effect
    decision = interrupt(reason="manager_approval", context={...})
    return {"charge_id": receipt["id"], "approved": decision.approved}
```

First execution: `charge_customer` runs (counter → 1), then the chain
suspends. On `chain.resume(...)`: the node re-runs from the top,
`charge_customer` runs again (counter → 2), then `interrupt()` returns the
resume value. The customer was charged twice.

## The mitigation

```python
from fastaiagent import idempotent

@idempotent
def charge_customer(amount):
    return stripe.charge(amount=amount)
```

Now the first call inside an execution stores its receipt. The second
call (after resume) finds the cache row and returns it without hitting
Stripe. Counter stays at 1.

## How keys work

The default key is `sha256(json.dumps([qualname, args, kwargs], sort_keys=True))`.
That works for primitive args. For live objects (DB sessions, Pydantic
models passed by reference, etc.) pass a custom `key_fn`:

```python
@idempotent(key_fn=lambda user, req: f"{user.id}:{req.id}")
def process(user, req):
    ...
```

## Scoping rules

- Same `execution_id` + same key → cache hit, body skipped.
- Different `execution_id` → cache miss, body runs again. The cache is
  scoped per execution; there is no cross-execution sharing.
- No active `execution_id` (called outside any chain run) → pass-through,
  body runs every time. Unit tests of `@idempotent` functions don't need
  any setup.

## Result serialization

The cached value goes through `pydantic_core.to_jsonable_python`, so
Pydantic models, dataclasses, datetimes, UUIDs, etc. round-trip cleanly.
Non-JSON-serializable returns raise `IdempotencyError` at the first call:

```python
@idempotent
def bad():
    return open("/tmp/x")  # file handles aren't JSONable

# Calling bad() inside a chain run raises:
# IdempotencyError: @idempotent function 'bad' returned a
# non-JSON-serializable value of type 'TextIOWrapper': ...
```

A consequence: the first call returns the original Python object, but a
**cache hit** returns the deserialized JSON form (a dict, list, or
primitive). Design idempotent functions to return plain data, or hydrate
the cached dict back into a model at the call site.

## Cache lifecycle

Idempotency rows live in the `idempotency_cache` table next to
`checkpoints` in `local.db`. To clean up after long-completed executions,
run `Checkpointer.prune(older_than=timedelta(days=7))` — this deletes
old `completed`/`failed` checkpoints and **all** idempotency rows older
than the cutoff. Suspended (`interrupted`) checkpoints are preserved so
pending HITL workflows survive the sweep.

---

## Next Steps

- [Human-in-the-Loop](hitl.md) — `interrupt()` and `Resume`, the primary
  reason `@idempotent` exists.
- [Checkpointing](checkpointing.md) — how chain state is persisted.
