"""
Tools — Functions the support agent can invoke.

Each tool receives RunContext[Deps] for dependency injection.
Tools are pure functions — no global state.
"""

import os

import fastaiagent as fa
from context import Deps

# ─── Knowledge Base ──────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

kb = fa.LocalKB(
    name="support-kb",
    path=os.path.join(_SCRIPT_DIR, ".fastaiagent-kb", "support-kb"),
    chunk_size=512,
    chunk_overlap=50,
)

_kb_ready = False


def _ensure_kb():
    global _kb_ready
    if not _kb_ready:
        kb_dir = os.path.join(_SCRIPT_DIR, "knowledge")
        for fname in sorted(os.listdir(kb_dir)):
            fpath = os.path.join(kb_dir, fname)
            if os.path.isfile(fpath):
                kb.add(fpath)
        _kb_ready = True


@fa.tool()
async def search_kb(query: str, ctx: fa.RunContext[Deps]) -> str:
    """Search the support knowledge base for product information, FAQs, and company policies.
    Always use this tool before answering product or policy questions."""
    _ensure_kb()
    results = kb.search(query, top_k=3)
    if not results:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join(
        f"**{r.chunk.metadata.get('source', 'KB')}** (relevance: {r.score:.2f})\n{r.chunk.content}"
        for r in results
    )


# ─── Ticket Creation ────────────────────────────────────────────────────────

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
    ticket = await ctx.state.ticket_client.create(
        subject=subject,
        description=f"Customer: {ctx.state.user_email}\n\n{description}",
        priority=priority,
    )
    return (
        f"Ticket created successfully.\n"
        f"  Ticket ID: {ticket['ticket_id']}\n"
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
