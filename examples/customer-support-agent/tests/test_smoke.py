"""
Smoke tests — no live LLM calls.

Exercise the deterministic parts of the example so a developer iterating on
prompts / tools gets fast feedback before they spend tokens on
``eval_suite.py`` or live runs.

Run from the example directory:

    pytest tests/

What's covered:
  * top-level imports of every module the example exposes
  * Agent construction (catches API drift in fa.Agent kwargs)
  * Tools standalone — search_kb against the real LocalKB; mock CRM/Ticket/Order
  * @idempotent ticket-id allocator returns deterministic ids per (email, subject, priority)
  * Guardrail factory functions return Guardrail instances at the expected positions
  * EvalSuite test cases parse and reach the LLMJudge constructor

What's NOT covered (by design):
  * full agent.arun loop — that's eval_suite.py's job; needs OPENAI_API_KEY
  * astream tokenization — same
  * UI HTTP endpoints — exercised manually
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make ``import agent / context / tools / ...`` work whether pytest is run
# from the example dir or from the repo root.
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ─── Module imports ──────────────────────────────────────────────────────────


def test_imports() -> None:
    import agent  # noqa: F401
    import context  # noqa: F401
    import guardrails  # noqa: F401
    import tools  # noqa: F401


# ─── Agent shape ─────────────────────────────────────────────────────────────


def test_agent_is_well_formed() -> None:
    from agent import agent

    assert agent.name == "customer-support"
    # Tools the support agent must expose to the LLM.
    tool_names = {t.name for t in agent.tools}
    assert {"search_kb", "create_ticket", "lookup_account", "check_order_status"} <= tool_names
    # Wired up the v1.6.0 surface this template advertises.
    assert agent.memory is not None, "memory must be wired for multi-turn REPL"
    assert agent._checkpointer is not None, "checkpointer must be wired for HITL"
    assert len(agent.middleware) >= 2, "ToolBudget + TrimLongMessages expected"
    assert len(agent.guardrails) == 2, "pii_filter + toxicity_check expected"


# ─── Tools (no LLM) ─────────────────────────────────────────────────────────


def test_search_kb_finds_refund_policy() -> None:
    """search_kb is a regular tool — we can call its underlying function
    directly without an LLM in the loop. It should return KB content for
    queries the seed corpus covers."""
    from tools import kb

    results = kb.search("refund policy", top_k=3)
    assert results, "KB should have ingested the seed knowledge files"
    text = " ".join(r.chunk.content for r in results).lower()
    assert "refund" in text


def test_kb_status_after_ingest() -> None:
    from tools import kb

    status = kb.status()
    assert status.get("chunk_count", 0) > 0


def test_mock_crm_lookup() -> None:
    from context import CRMClient

    client = CRMClient()

    async def _go() -> dict | None:
        return await client.lookup("alice@acme.com")

    account = asyncio.run(_go())
    assert account is not None
    assert account["plan"] == "Enterprise"


def test_mock_order_status() -> None:
    from context import OrderClient

    client = OrderClient()

    async def _go() -> dict | None:
        return await client.check_status("ORD-9912")

    order = asyncio.run(_go())
    assert order is not None
    assert order["status"] == "shipped"


# ─── Idempotent ticket-id allocator ──────────────────────────────────────────


def test_ticket_id_keying_is_stable() -> None:
    """``_ticket_idem_key`` must produce the same key for the same logical
    ticket even if called from different processes — that's the whole point
    of @idempotent in the resume / replay flow."""
    from tools import _ticket_idem_key

    k1 = _ticket_idem_key(user_email="x@y.com", subject="Login broken", priority="high")
    k2 = _ticket_idem_key(user_email="x@y.com", subject="Login broken", priority="high")
    k3 = _ticket_idem_key(user_email="x@y.com", subject="Login broken", priority="urgent")
    k4 = _ticket_idem_key(user_email="z@y.com", subject="Login broken", priority="high")
    assert k1 == k2
    assert k1 != k3, "priority must affect the key"
    assert k1 != k4, "user_email must affect the key"


def test_allocate_ticket_id_outside_chain_runs_inline() -> None:
    """Without an active execution_id / checkpointer, @idempotent falls
    through to the wrapped function. The example's helper should still
    return a well-formed dict (the test for cache *reuse* needs a chain
    run with a checkpointer — covered by the eval/replay flows)."""
    from tools import _allocate_ticket_id

    result = _allocate_ticket_id(user_email="x@y.com", subject="x", priority="low")
    assert result["ticket_id"].startswith("TKT-")
    assert result["user_email"] == "x@y.com"


# ─── Guardrails ─────────────────────────────────────────────────────────────


def test_guardrails_at_expected_positions() -> None:
    from fastaiagent.guardrail import GuardrailPosition

    from guardrails import pii_filter, toxicity_check

    assert pii_filter.position == GuardrailPosition.output
    assert toxicity_check.position == GuardrailPosition.input


# ─── Eval scorers + cases parse ──────────────────────────────────────────────


def test_eval_cases_well_formed() -> None:
    from eval_suite import EVAL_CASES, correctness, helpfulness, safety

    assert len(EVAL_CASES) >= 5
    for case in EVAL_CASES:
        assert "input" in case and "expected" in case

    # Each LLMJudge has a name / criteria / scale — runtime ctor sanity.
    for scorer in (correctness, helpfulness, safety):
        assert scorer.name in {"correctness", "helpfulness", "safety"}
