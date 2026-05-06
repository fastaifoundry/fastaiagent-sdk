"""Smoke tests — no live LLM calls.

Exercise the deterministic parts of the example so a developer iterating on
the workflow + ICP rubric gets fast feedback before they spend tokens on
``eval_suite.py`` or live runs.

Run from the example directory:

    python -m pytest tests/

Coverage:
  * imports of every example module
  * Chain topology — node ids, edge conditions, conditional routing
  * Mock enrichment + idempotent _persist_send
  * Pluggable backend selection via env vars
  * Tools work standalone (with strict ctx required, post-bug-fix)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import fastaiagent as fa

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ─── Module imports ──────────────────────────────────────────────────────────


def test_imports() -> None:
    import agent  # noqa: F401
    import eval_suite  # noqa: F401
    import replay_demo  # noqa: F401
    import streaming_demo  # noqa: F401
    import tools  # noqa: F401
    import workflow  # noqa: F401


# ─── Chain topology ─────────────────────────────────────────────────────────


def test_chain_topology() -> None:
    from workflow import build_chain

    chain = build_chain()
    assert chain.name == "sales-sdr"
    node_ids = [n.id for n in chain.nodes]
    assert node_ids == ["enrich", "score", "draft", "send", "log_crm", "disqualify"]

    # Edges — pin the conditional routing logic.
    edges = {(e.source, e.target): e for e in chain.edges}
    assert ("enrich", "score") in edges
    assert ("score", "draft") in edges
    assert ("score", "disqualify") in edges
    assert ("draft", "send") in edges
    assert ("send", "log_crm") in edges

    # Score → draft uses a `>= threshold` condition; disqualify uses `<`.
    assert "{{state.output.score}} >=" in (edges[("score", "draft")].condition or "")
    assert "{{state.output.score}} <" in (edges[("score", "disqualify")].condition or "")


# ─── Tools ──────────────────────────────────────────────────────────────────


def test_enrich_lead_finds_known_prospect() -> None:
    from tools import SDRDeps, enrich_lead

    ctx = fa.RunContext(state=SDRDeps())
    # FunctionTool wraps the body — call via aexecute to exercise the
    # @fa.tool() decorator path. asyncio.run handles the coroutine.
    import asyncio

    result = asyncio.run(enrich_lead.aexecute({"prospect_email": "alice@acme-saas.com"}, context=ctx))
    assert result.error is None
    assert result.output["found"] is True
    assert result.output["company"] == "Acme SaaS"


def test_enrich_lead_missing_returns_not_found() -> None:
    from tools import SDRDeps, enrich_lead

    ctx = fa.RunContext(state=SDRDeps())
    import asyncio

    result = asyncio.run(enrich_lead.aexecute({"prospect_email": "ghost@nowhere.test"}, context=ctx))
    assert result.error is None
    assert result.output["found"] is False


def test_idempotent_send_returns_dict() -> None:
    """``_persist_send`` is decorated with ``@idempotent``. Outside a chain
    run, the decorator falls through to the wrapped function and we get a
    well-formed receipt dict back."""
    from tools import _persist_send

    receipt = _persist_send(to="x@y.com", subject="subj", body="hello")
    assert receipt["sent"] is True
    assert receipt["msg_id"].startswith("MSG-")


def test_email_idempotency_key_stable_across_calls() -> None:
    from tools import _email_idem_key

    k1 = _email_idem_key(to="x@y.com", subject="s", body="b")
    k2 = _email_idem_key(to="x@y.com", subject="s", body="b")
    k3 = _email_idem_key(to="x@y.com", subject="s", body="b!")
    assert k1 == k2
    assert k1 != k3, "body change must alter the cache key"


# ─── Backend selection ──────────────────────────────────────────────────────


def test_default_backend_is_mock() -> None:
    from tools import _ENRICHMENT_BACKENDS

    assert "mock" in _ENRICHMENT_BACKENDS
    assert "clearbit" in _ENRICHMENT_BACKENDS


def test_unknown_backend_raises_via_real_call(monkeypatch) -> None:
    """Setting CLEARBIT to backend without the API key raises a friendly
    runtime error rather than a cryptic httpx exception."""
    from tools import _enrich_clearbit

    monkeypatch.delenv("CLEARBIT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CLEARBIT_API_KEY"):
        _enrich_clearbit("anyone@anywhere.com")


# ─── Eval cases shape ───────────────────────────────────────────────────────


def test_eval_cases_well_formed() -> None:
    from eval_suite import EVAL_CASES

    assert len(EVAL_CASES) >= 4
    for case in EVAL_CASES:
        assert "input" in case
        assert "expected_qualified" in case
        assert isinstance(case["expected_qualified"], bool)
        assert isinstance(case.get("must_mention", []), list)
