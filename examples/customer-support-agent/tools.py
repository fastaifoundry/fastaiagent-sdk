"""
Tools — Functions the support agent can invoke.

Each tool receives RunContext[Deps] for dependency injection.
Tools are pure functions — no global state beyond the module-level KB singleton.

This file demonstrates two newer SDK features:

  * ``interrupt()`` (v1.0) inside ``create_ticket`` for human-in-the-loop
    approval on high-priority and billing tickets. The Agent's
    ``SQLiteCheckpointer`` persists the suspension; ``agent.aresume(...)``
    re-enters the tool with the human's decision.
  * ``@idempotent`` (v1.0) on a synchronous helper that allocates the
    ticket id. If the agent loop is replayed (resume after interrupt, crash
    recovery, ``Replay.fork_at`` rerun), the same id is returned — no duplicate
    tickets get filed.
"""

import hashlib
import os

import fastaiagent as fa
from fastaiagent.chain.idempotent import idempotent

from context import Deps

# ─── Knowledge Base ──────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Persistent KB with hybrid search (FAISS + BM25).
# On first run: ingests docs and persists to SQLite.
# On subsequent runs: loads instantly from disk — no re-embedding.
kb = fa.LocalKB(
    name="support-kb",
    path=os.path.join(_SCRIPT_DIR, ".fastaiagent-kb"),
    chunk_size=512,
    chunk_overlap=50,
)

# Ingest knowledge files if KB is empty (first run or after clear)
if kb.status()["chunk_count"] == 0:
    kb.add(os.path.join(_SCRIPT_DIR, "knowledge"))


@fa.tool()
async def search_kb(query: str, ctx: fa.RunContext[Deps]) -> str:
    """Search the support knowledge base for product information, FAQs, and company policies.
    Always use this tool before answering product or policy questions."""
    results = kb.search(query, top_k=3)
    if not results:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join(
        f"**{r.chunk.metadata.get('source', 'KB')}** (relevance: {r.score:.2f})\n{r.chunk.content}"
        for r in results
    )


# ─── Ticket Creation (HITL + idempotent) ─────────────────────────────────────


def _ticket_idem_key(*, user_email: str, subject: str, priority: str) -> str:
    """Stable cache key for an idempotent ticket allocation.

    The default key builder JSON-encodes args with ``default=str``, which would
    embed ``RunContext`` and ``TicketClient`` repr strings — unstable across
    processes. We pin the key to the human-meaningful tuple instead.
    """
    digest = hashlib.sha256(f"{user_email}|{subject}|{priority}".encode()).hexdigest()[:16]
    return f"ticket:{digest}"


@idempotent(key_fn=_ticket_idem_key)
def _allocate_ticket_id(*, user_email: str, subject: str, priority: str) -> dict:
    """Allocate a stable ticket id. ``@idempotent`` ensures the agent's
    checkpointer caches this allocation under ``execution_id`` — so a resume
    after an ``interrupt()`` (or a ``Replay.fork_at`` rerun) reuses the same
    ticket id rather than minting a new one."""
    import time
    return {
        "ticket_id": f"TKT-{int(time.time() * 1000)}",
        "user_email": user_email,
        "subject": subject,
        "priority": priority,
    }


@fa.tool()
async def create_ticket(
    subject: str,
    description: str,
    priority: str,
    ctx: fa.RunContext[Deps],
) -> str:
    """Create a support ticket for issues that cannot be resolved directly.
    Priority should be 'low', 'medium', 'high', or 'urgent'.
    Use this for billing disputes, refund requests, or complex technical issues."""
    user_email = ctx.state.user_email

    # HITL: high-impact tickets require human approval before they're filed.
    # The Agent's SQLiteCheckpointer persists the suspension; the REPL prompts
    # the user, then calls ``agent.aresume(...)`` with the decision.
    if priority in ("high", "urgent") or "billing" in subject.lower() or "refund" in description.lower():
        decision = fa.interrupt(
            reason="ticket_approval_required",
            context={
                "subject": subject,
                "priority": priority,
                "user_email": user_email,
                "reason_for_review": "high-impact / billing / refund",
            },
        )
        if not decision.approved:
            return (
                f"Ticket creation declined by reviewer "
                f"({decision.metadata.get('approver', 'unknown')}). "
                "I've logged your request and a human agent will reach out directly."
            )

    # Allocate ticket id idempotently (survives resume/replay).
    allocation = _allocate_ticket_id(
        user_email=user_email, subject=subject, priority=priority
    )
    ticket = await ctx.state.ticket_client.create(
        subject=subject,
        description=f"Customer: {user_email}\n\n{description}",
        priority=priority,
    )
    return (
        f"Ticket created successfully.\n"
        f"  Ticket ID: {allocation['ticket_id']}\n"
        f"  Backend ref: {ticket['ticket_id']}\n"
        f"  Subject: {ticket['subject']}\n"
        f"  Priority: {ticket['priority']}\n"
        f"  Status: {ticket['status']}\n\n"
        f"The customer will receive an email confirmation."
    )


# ─── Account Lookup ─────────────────────────────────────────────────────────


@fa.tool()
async def lookup_account(email: str, ctx: fa.RunContext[Deps]) -> str:
    """Look up a customer's account information by email address.
    Use this to check account status, plan details, or verify identity."""
    account = await ctx.state.crm_client.lookup(email)
    if not account:
        return f"No account found for {email}."
    return (
        f"Account found:\n"
        f"  Name: {account['name']}\n"
        f"  Company: {account['company']}\n"
        f"  Plan: {account['plan']}\n"
        f"  Account ID: {account['account_id']}\n"
        f"  Status: {account['status']}"
    )


# ─── Order Status ───────────────────────────────────────────────────────────


@fa.tool()
async def check_order_status(order_id: str, ctx: fa.RunContext[Deps]) -> str:
    """Check the status of a customer's order by order ID.
    Returns shipping status, tracking number, and estimated delivery."""
    order = await ctx.state.order_client.check_status(order_id)
    if not order:
        return f"No order found with ID {order_id}. Please verify the order number."
    parts = [f"Order {order_id}:", f"  Status: {order['status']}"]
    if order.get("tracking"):
        parts.append(f"  Tracking: {order['tracking']}")
    if order.get("eta"):
        parts.append(f"  ETA: {order['eta']}")
    return "\n".join(parts)
