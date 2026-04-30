# Side effects & idempotency

This is the page every team adding durability has to read once. It's also
the easiest one to skip. Read it once.

## The footgun

A node that pauses re-runs from the top on resume. Anything that node
did *before* it called `interrupt()` runs again.

```python
def approval_node(state):
    receipt = charge_customer(state["amount"])  # ← side effect
    decision = interrupt(reason="manager_approval", context={...})
    return {"charge_id": receipt["id"], "approved": decision.approved}
```

| Run | What happens |
|---|---|
| First execute | `charge_customer` runs (counter → 1). Chain suspends. |
| `chain.aresume(...)` | The node re-runs from the top. `charge_customer` runs **again** (counter → 2). Then `interrupt()` returns the resume value. |

The customer gets charged twice. This is the same shape that bites every
durable-workflow framework — Temporal, Inngest, Trigger.dev all warn
about it. The fix is the same in all of them: tell the framework "this
function is replay-safe, don't re-execute on rerun."

## The mitigation: `@idempotent`

```python
from fastaiagent import idempotent

@idempotent
def charge_customer(amount: int):
    return stripe.charge(amount=amount)
```

| Run | What happens |
|---|---|
| First execute | `charge_customer` body runs. Receipt stored in `idempotency_cache(execution_id, key)`. Chain suspends. |
| `chain.aresume(...)` | Node re-runs. `charge_customer(...)` is called — **but the body is skipped**; the cached receipt is returned. `interrupt()` resumes. |

[`@idempotent`](../chains/idempotency.md) keys on
`(execution_id, sha256(qualname + args + kwargs))`. The cache lives in
the same SQLite (or Postgres) the checkpointer uses, so it survives
process restarts.

## Where it bites — concrete examples

### Payment processing

```python
@idempotent
def charge_card(card_token: str, amount_cents: int):
    return stripe.charge(amount=amount_cents, source=card_token)
```

Without `@idempotent`: a `chain.aresume(...)` after suspend re-charges
the card. Two `stripe.Charge` records, two settled debits, one very
unhappy customer.

With `@idempotent`: the second call returns the cached `stripe.Charge`
dict. Stripe is never re-hit.

If you can't decorate the function (3rd-party SDK, can't add a wrapper),
use Stripe's own idempotency key — and pass a deterministic key derived
from `execution_id`:

```python
from fastaiagent.chain.interrupt import _execution_id

def charge_card(amount_cents):
    exec_id = _execution_id.get() or "live"  # falls back outside a chain
    return stripe.Charge.create(
        amount=amount_cents,
        idempotency_key=f"refund:{exec_id}",
    )
```

### Database writes

```python
# Bad: re-executes the INSERT on resume.
def record_invoice(state):
    db.execute("INSERT INTO invoices VALUES (?, ?)", (state["id"], state["amount"]))
```

Two fixes:

```python
# Option A — INSERT ... ON CONFLICT DO NOTHING
def record_invoice(state):
    db.execute(
        "INSERT INTO invoices VALUES (?, ?) ON CONFLICT (id) DO NOTHING",
        (state["id"], state["amount"]),
    )

# Option B — wrap the call with @idempotent
@idempotent
def record_invoice(invoice_id, amount):
    db.execute("INSERT INTO invoices VALUES (?, ?)", (invoice_id, amount))
```

Option A is preferred when the DB itself enforces the invariant; Option
B when the call has no natural conflict key.

### Email / Slack / webhook sends

```python
@idempotent
def send_confirmation(to: str, subject: str, body: str):
    return ses.send_email(Destination={"ToAddresses": [to]}, Message={...})
```

Most providers accept a client-generated idempotency key; pass one when
you can. `@idempotent` is the catch-all when you can't.

### Cloud resource creation

```python
# Without protection: every resume creates ANOTHER bucket.
@idempotent
def create_bucket(name: str, region: str):
    return s3.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": region})
```

Or use deterministic naming so the second `create_bucket` is a
benign-409 instead of a leak.

### Non-deterministic LLM calls

When a turn-boundary checkpoint resumes, the executor re-issues the LLM
call. With `temperature=0.7`, the second response will be slightly
different from the first. If the chain's branching logic depends on the
LLM's exact output, the resumed run may diverge.

Two fixes:

1. **Lower temperature** for the deciding step. A summarizer at 0.0
   produces deterministic output. (Cheap.)
2. **Cache the LLM call** — wrap your LLM client with `@idempotent` on
   `(messages, temperature, model)`. Returned content is cached for the
   execution; resumes use the cached response. (Expensive in serialization,
   but bulletproof.)

### Time-dependent decisions

```python
def expire_voucher(state):
    now = datetime.now(timezone.utc)         # ← changes on resume!
    if now > state["voucher_expires_at"]:
        return {"valid": False}
    return {"valid": True}
```

If the human approves an hour after pause, `now` is an hour later than
when the chain originally checked. Two fixes:

1. **Capture the timestamp at first execution** in `state` and read it
   on resume:

   ```python
   def expire_voucher(state):
       check_at = state.get("checked_at") or datetime.now(timezone.utc)
       state["checked_at"] = check_at  # persisted in the next checkpoint
       if check_at > state["voucher_expires_at"]:
           return {"valid": False}
       return {"valid": True}
   ```

2. **Move the time read into a `@idempotent` function**:

   ```python
   @idempotent
   def now_utc():
       return datetime.now(timezone.utc).isoformat()
   ```

   The first call records the timestamp; subsequent calls in the same
   execution return the recorded value.

## The frozen-context invariant

The `context` dict you pass to `interrupt()` is **JSON-serialized at
suspend time** and never recomputed. The human approves a specific
snapshot.

```python
decision = interrupt(
    reason="manager_approval",
    context={"amount": state["amount"], "balance": current_balance()},
)
```

If the customer's balance changes between pause and resume, the resumer
sees the **original** balance — the one the approver looked at. If your
code needs the *current* balance when it resumes, read it again from
`state` (which is also rehydrated on resume) or from the live system —
do not rely on the frozen `context`.

Why this is the right default: a human approver looks at a specific
context, makes a decision, clicks Approve. If the context could shift
under their feet, the audit log lies. We chose correctness of audit over
freshness of data.

## Choosing a strategy

| Side effect | Best fit |
|---|---|
| Idempotent by nature (e.g. PUT /resource/123) | Nothing needed. |
| Non-idempotent but has provider idempotency keys (Stripe, SES, S3) | Pass a deterministic key derived from `execution_id`. |
| Non-idempotent and no provider idempotency keys | `@idempotent`. |
| Non-deterministic decision (LLM, time, random) | Capture the result in state, or wrap with `@idempotent`. |
| Multi-statement DB write that should be all-or-nothing | Make the function atomic with a transaction; then `@idempotent` it. |

## Cleanup

Idempotency rows accumulate. The
[`Checkpointer.prune(older_than=…)`](api-reference.md#checkpointer-protocol)
method clears `idempotency_cache` rows and completed/failed checkpoints
older than the cutoff. **Suspended (`interrupted`) checkpoints are
preserved** — pruning them would orphan an active human-in-the-loop. A
weekly cron is the typical shape:

```python
from datetime import timedelta
from fastaiagent import SQLiteCheckpointer

cp = SQLiteCheckpointer()
cp.setup()
cp.prune(older_than=timedelta(days=7))
```

## See also

- [Idempotency reference](../chains/idempotency.md) — full `@idempotent`
  semantics, key functions, serialization rules.
- [Patterns — payment processing](patterns.md#payment-processing) — a
  worked example combining `interrupt()` + `@idempotent`.
