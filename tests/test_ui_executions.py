"""Integration tests for the v1.0 ``/api/executions`` HTTP surface (Phase 9).

Three endpoints under test:

    GET  /api/executions/{execution_id}
    GET  /api/pending-interrupts
    POST /api/executions/{execution_id}/resume

Plus a runners-registry validation test on ``build_app(runners=...)``.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent import (  # noqa: E402
    Chain,
    FunctionTool,
    Resume,
    SQLiteCheckpointer,
    interrupt,
)
from fastaiagent.chain.node import NodeType  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402

# ---------- Helpers — build a chain that interrupts on first run --------


def _approval_fn(amount: str) -> dict[str, Any]:
    n = int(amount)
    if n > 1000:
        decision = interrupt(
            reason="manager_approval",
            context={"amount": n, "policy": "high-value"},
        )
        return {"approved": decision.approved, "approver": decision.metadata.get("approver")}
    return {"approved": True, "auto": True}


def _final_fn(approved: str) -> dict[str, Any]:
    is_approved = str(approved).lower() in ("true", "1", "yes")
    return {"final_approved": is_approved, "step_final_done": True}


def _build_chain(ckpt_db: str) -> Chain:
    store = SQLiteCheckpointer(db_path=ckpt_db)
    chain = Chain(
        "ui-exec-test",
        checkpoint_enabled=True,
        checkpointer=store,
    )
    chain.add_node(
        "approval",
        tool=FunctionTool(name="approval_tool", fn=_approval_fn),
        type=NodeType.tool,
        input_mapping={"amount": "{{state.amount}}"},
    )
    chain.add_node(
        "final",
        tool=FunctionTool(name="final_tool", fn=_final_fn),
        type=NodeType.tool,
        input_mapping={"approved": "{{state.output.approved}}"},
    )
    chain.connect("approval", "final")
    return chain


@pytest.fixture
def app_with_chain(temp_dir: Path) -> tuple[TestClient, Chain, str]:
    db_path = str(temp_dir / "exec-routes.db")
    chain = _build_chain(db_path)
    execution_id = f"ui-{uuid.uuid4().hex[:8]}"
    paused = chain.execute(
        {"amount": 50_000},
        execution_id=execution_id,
    )
    assert paused.status == "paused"

    app = build_app(db_path=db_path, no_auth=True, runners=[chain])
    return TestClient(app), chain, execution_id


# ---------- GET /api/executions/{id} ------------------------------------


def test_get_execution_returns_history_and_status(
    app_with_chain: tuple[TestClient, Chain, str],
) -> None:
    client, _chain, execution_id = app_with_chain
    r = client.get(f"/api/executions/{execution_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["execution_id"] == execution_id
    assert body["chain_name"] == "ui-exec-test"
    assert body["status"] == "interrupted"
    assert body["checkpoint_count"] >= 1
    nodes = [cp["node_id"] for cp in body["checkpoints"]]
    assert "approval" in nodes


def test_get_execution_404_for_unknown_id(app_with_chain: tuple[TestClient, Chain, str]) -> None:
    client, _chain, _execution_id = app_with_chain
    r = client.get("/api/executions/does-not-exist")
    assert r.status_code == 404


# ---------- GET /api/pending-interrupts ---------------------------------


def test_pending_interrupts_lists_the_paused_chain(
    app_with_chain: tuple[TestClient, Chain, str],
) -> None:
    client, _chain, execution_id = app_with_chain
    r = client.get("/api/pending-interrupts")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] >= 1
    rows = [row for row in body["items"] if row["execution_id"] == execution_id]
    assert len(rows) == 1
    row = rows[0]
    assert row["reason"] == "manager_approval"
    assert row["context"] == {"amount": 50_000, "policy": "high-value"}


# ---------- POST /api/executions/{id}/resume ----------------------------


def test_resume_completes_the_chain(app_with_chain: tuple[TestClient, Chain, str]) -> None:
    client, _chain, execution_id = app_with_chain
    r = client.post(
        f"/api/executions/{execution_id}/resume",
        json={"approved": True, "metadata": {"approver": "alice"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["execution_id"] == execution_id
    assert body["chain_name"] == "ui-exec-test"
    result = body["result"]
    assert result["status"] == "completed"

    # Pending row was atomically claimed — list-pending should not include it.
    r2 = client.get("/api/pending-interrupts")
    assert r2.status_code == 200
    remaining = [row for row in r2.json()["items"] if row["execution_id"] == execution_id]
    assert remaining == []


def test_double_resume_returns_409(app_with_chain: tuple[TestClient, Chain, str]) -> None:
    client, _chain, execution_id = app_with_chain
    # First resume succeeds.
    r1 = client.post(
        f"/api/executions/{execution_id}/resume",
        json={"approved": True, "metadata": {}},
    )
    assert r1.status_code == 200

    # Second resume — pending row already claimed → 409 Conflict.
    r2 = client.post(
        f"/api/executions/{execution_id}/resume",
        json={"approved": False, "metadata": {}},
    )
    assert r2.status_code == 409, r2.text
    assert "already resumed" in r2.json()["detail"].lower()


def test_resume_503_when_no_runner_registered(temp_dir: Path) -> None:
    db_path = str(temp_dir / "no-runner.db")
    chain = _build_chain(db_path)
    execution_id = f"ui-norunner-{uuid.uuid4().hex[:8]}"
    paused = chain.execute({"amount": 50_000}, execution_id=execution_id)
    assert paused.status == "paused"

    # Build the app WITHOUT registering the chain.
    app = build_app(db_path=db_path, no_auth=True)
    client = TestClient(app)

    r = client.post(
        f"/api/executions/{execution_id}/resume",
        json={"approved": True, "metadata": {}},
    )
    assert r.status_code == 503, r.text
    detail = r.json()["detail"].lower()
    assert "no runner registered" in detail
    assert "build_app" in detail


def test_resume_404_when_unknown_execution(app_with_chain: tuple[TestClient, Chain, str]) -> None:
    client, _chain, _execution_id = app_with_chain
    r = client.post(
        "/api/executions/does-not-exist/resume",
        json={"approved": True, "metadata": {}},
    )
    assert r.status_code == 404


# ---------- build_app(runners=...) validation ---------------------------


class _NotARunner:
    """Object that lacks the .aresume() method."""

    name = "not-a-runner"


def test_build_app_rejects_runners_without_aresume(temp_dir: Path) -> None:
    db_path = str(temp_dir / "x.db")
    with pytest.raises(ValueError, match="aresume"):
        build_app(db_path=db_path, no_auth=True, runners=[_NotARunner()])


def test_resume_uses_explicit_resume_value_metadata(
    app_with_chain: tuple[TestClient, Chain, str],
) -> None:
    client, _chain, execution_id = app_with_chain
    r = client.post(
        f"/api/executions/{execution_id}/resume",
        json={"approved": True, "metadata": {"approver": "bob"}},
    )
    assert r.status_code == 200
    body = r.json()
    # ``node_results`` preserves per-node returns even after the next tool's
    # output overwrites ``state.output``. The approval node's return must
    # carry the resumer's metadata through to ``approver``.
    approval_result = body["result"]["node_results"]["approval"]
    inner = approval_result["output"]
    assert inner["approver"] == "bob"
    assert inner["approved"] is True
    # And state.output reflects the final tool's outcome.
    final_state = body["result"]["final_state"]
    assert final_state["output"]["final_approved"] is True


# ---------- _build_chain idempotence (sanity) ---------------------------


def test_resume_via_chain_python_api_matches_http_result(temp_dir: Path) -> None:
    """Sanity: resuming via Python API yields the same result shape as HTTP.

    Confirms the HTTP endpoint is a thin pass-through over the same
    ``aresume`` call the user would make directly.
    """
    from fastaiagent._internal.async_utils import run_sync

    db_path_a = str(temp_dir / "py-resume.db")
    chain_a = _build_chain(db_path_a)
    exec_a = f"py-{uuid.uuid4().hex[:8]}"
    paused = chain_a.execute({"amount": 50_000}, execution_id=exec_a)
    assert paused.status == "paused"
    py_result = run_sync(
        chain_a.resume(exec_a, resume_value=Resume(approved=True, metadata={"approver": "alice"}))
    )
    assert py_result.status == "completed"

    # HTTP path
    db_path_b = str(temp_dir / "http-resume.db")
    chain_b = _build_chain(db_path_b)
    exec_b = f"http-{uuid.uuid4().hex[:8]}"
    paused_b = chain_b.execute({"amount": 50_000}, execution_id=exec_b)
    assert paused_b.status == "paused"
    app = build_app(db_path=db_path_b, no_auth=True, runners=[chain_b])
    client = TestClient(app)
    http_body = client.post(
        f"/api/executions/{exec_b}/resume",
        json={"approved": True, "metadata": {"approver": "alice"}},
    ).json()

    # Both paths converge on the same status + final tool output.
    assert http_body["result"]["status"] == "completed"
    py_final = py_result.final_state["output"]
    http_final = http_body["result"]["final_state"]["output"]
    assert py_final.get("final_approved") == http_final.get("final_approved")
    assert py_final.get("step_final_done") is True
    assert http_final.get("step_final_done") is True
