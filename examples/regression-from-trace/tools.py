"""Two versions of ``lookup_order``: the buggy original and the fix.

The bug:
    ``lookup_order_buggy`` falls back to the first known order
    (``ORD-001``) when the requested ID isn't found, returning a
    *plausible-but-wrong* record. The LLM has no signal that the
    lookup actually failed, so it confidently reports ORD-001's
    details as if they belonged to the user's order — exactly the
    kind of silent failure that's hardest to catch in production.

The fix:
    ``lookup_order_fixed`` returns a structured ``{"error": ...}``
    dict so "not found" is unambiguous. The LLM sees the error and
    replies "I couldn't find order ORD-999".

Both tools share the same ``KNOWN_ORDERS`` table. Swap them via
``ForkedReplay.with_tool_override("lookup_order", fixed)`` to see the
trace → replay → fix → regression loop end-to-end.
"""

from __future__ import annotations

from fastaiagent import FunctionTool

KNOWN_ORDERS: dict[str, dict[str, str]] = {
    "ORD-001": {
        "id": "ORD-001",
        "item": "MacBook Pro 16-inch",
        "status": "delivered",
        "delivered_on": "2026-04-03",
    },
    "ORD-002": {
        "id": "ORD-002",
        "item": "AirPods Pro",
        "status": "processing",
        "estimated_ship": "2026-05-30",
    },
}

# Sentinel default the buggy fallback returns. Picked so the LLM has
# no way to recover — the data looks structurally valid.
_BUGGY_FALLBACK_ID = "ORD-001"


def _lookup_order_buggy(order_id: str) -> dict[str, str]:
    """Original buggy implementation.

    Silently falls back to ``KNOWN_ORDERS[_BUGGY_FALLBACK_ID]`` when
    the requested ``order_id`` isn't in the table, and stamps the
    requested ID onto the returned record so the response looks
    completely valid to the LLM. From the model's perspective this is
    indistinguishable from a successful lookup — the failure is
    invisible until a human notices the agent is confidently shipping
    wrong details to customers.
    """
    found = KNOWN_ORDERS.get(order_id)
    if found is not None:
        return found
    # Silent fallback — overwrite the id field so the response looks
    # coherent and the LLM has nothing to cross-check against.
    fallback = dict(KNOWN_ORDERS[_BUGGY_FALLBACK_ID])
    fallback["id"] = order_id
    return fallback


def _lookup_order_fixed(order_id: str) -> dict[str, str]:
    """Fixed implementation.

    Returns a structured ``{"error": ...}`` dict when the order ID
    isn't found, giving the LLM an unambiguous "not found" signal to
    reason about.
    """
    found = KNOWN_ORDERS.get(order_id)
    if found is None:
        return {"error": f"order {order_id} not found"}
    return found


def buggy_lookup_order_tool() -> FunctionTool:
    """The tool to ship in ``capture.py`` — produces the failing trace."""
    return FunctionTool(name="lookup_order", fn=_lookup_order_buggy)


def fixed_lookup_order_tool() -> FunctionTool:
    """The tool to override with in ``fix.py``. Keep the same ``name=``
    so :meth:`ForkedReplay.with_tool_override` swaps it cleanly."""
    return FunctionTool(name="lookup_order", fn=_lookup_order_fixed)
