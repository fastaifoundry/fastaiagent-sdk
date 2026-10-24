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

`async def` functions work the same way (since 1.92.0): the body is awaited
once and its result cached.

```python
@idempotent
async def charge_customer(amount):
    return await stripe_async.charge(amount=amount)
```

!!! warning "Before 1.92.0"
    An `async def` wrapped in `@idempotent` never ran: the wrapper cached the
    un-awaited coroutine and raised. Inside an agent the tool reported a failure
    and the run still finished normally. A function returning `None` was also
    never cached, so it ran again on every call. Both are fixed; cached rows
    from older versions are still read.

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
- No active `execution_id` or no checkpointer (called outside a checkpointed
  Agent, Chain or Swarm run) → pass-through, body runs every time. Unit tests
  of `@idempotent` functions don't need any setup.
- A `None` result is cached like any other value.
- **A run nested in a checkpointed run shares its cache.** An Agent node of a
  Chain (or a Chain run from a tool) runs under an execution id of its own, but
  the outer run's resume re-runs it from the top, so its `@idempotent` calls are
  keyed by the outer run's execution and stored in the outer run's checkpointer.
  Before 1.92.0 the inner Agent replaced the cache with its own — usually none —
  and a charge inside an Agent node ran again on every Chain resume.

## Result serialization

The cached value goes through `pydantic_core.to_jsonable_python`, so
Pydantic models, dataclasses, datetimes, UUIDs, etc. round-trip cleanly.
Non-JSON-serializable returns raise `IdempotencyError` at the first call:

```python
@idempotent
def bad():
    return open("/tmp/x")  # file handles aren't JSONable

# IdempotencyError: @idempotent function 'bad' returned a
# non-JSON-serializable value of type 'TextIOWrapper': ...
```

!!! note "Inside a chain you catch `ToolExecutionError`, not `IdempotencyError`"
    The decorator raises `IdempotencyError`, but a chain reaches your function
    through a `FunctionTool`, and `FunctionTool` wraps **any** exception the
    function raises:

    ```
    ToolExecutionError: Tool 'n' failed: @idempotent function 'bad' returned a
    non-JSON-serializable value of type 'TextIOWrapper': ...
    ```

    The original is preserved on `__cause__`, so catch the wrapper and inspect
    it:

    ```python
    from fastaiagent import IdempotencyError
    from fastaiagent._internal.errors import ToolExecutionError

    try:
        chain.execute({"order_id": "A-1"})
    except ToolExecutionError as e:
        if isinstance(e.__cause__, IdempotencyError):
            ...  # a cached node returned something unserializable
    ```

    Called directly — in a unit test, or from a runtime that is not the chain
    executor — it raises `IdempotencyError` unwrapped.

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
