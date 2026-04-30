"""End-to-end quality gate — Local UI durability surface (v1.0 Phase 10).

Walks the same sequence the operator does in the browser:

    1. A chain calls ``interrupt()`` — pending row appears.
    2. ``GET /api/overview`` exposes ``pending_approvals_count``.
    3. ``GET /api/pending-interrupts`` lists the row that the
       ``/approvals`` page renders.
    4. ``GET /api/executions/<id>`` returns the checkpoint history that the
       ``/executions/:id`` page reads.
    5. ``POST /api/executions/<id>/resume`` (Approve) completes the chain.
    6. After resume: pending row is gone, overview counters update,
       and the resumed run is visible in the execution history.
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
    SQLiteCheckpointer,
    interrupt,
)
from fastaiagent.chain.node import NodeType  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402
from tests.e2e.conftest import require_env  # noqa: E402

pytestmark = pytest.mark.e2e


# ---------- Chain that pauses on first run ------------------------------


def _approval_fn(amount: str) -> dict[str, Any]:
    n = int(amount)
    if n > 1000:
        decision = interrupt(
            reason="manager_approval",
            context={"amount": n, "policy": "high-value"},
        )
        return {
            "approved": decision.approved,
            "approver": decision.metadata.get("approver"),
        }
    return {"approved": True, "auto": True}


def _final_fn(approved: str) -> dict[str, Any]:
    return {
        "final_approved": str(approved).lower() in ("true", "1", "yes"),
        "step_final_done": True,
    }


def _build_chain(ckpt_db: str) -> Chain:
    chain = Chain(
        "ui-durability-chain",
        checkpoint_enabled=True,
        checkpointer=SQLiteCheckpointer(db_path=ckpt_db),
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


# ---------- The end-to-end gate -----------------------------------------


class TestLocalUIDurabilityGate:
    """Approvals → Resume round-trip that the v1.0 UI bakes into screens."""

    def test_full_paused_to_resumed_cycle(self, tmp_path: Path) -> None:
        require_env()

        db_path = str(tmp_path / "ui-gate.db")
        chain = _build_chain(db_path)
        execution_id = f"ui-gate-{uuid.uuid4().hex[:8]}"

        # 1. Run the chain. interrupt() suspends.
        paused = chain.execute(
            {"amount": 50_000},
            execution_id=execution_id,
        )
        assert paused.status == "paused"

        # Build the API the UI talks to.
        app = build_app(db_path=db_path, no_auth=True, runners=[chain])
        client = TestClient(app)

        # 2. Overview KPIs reflect the suspended workflow.
        ov = client.get("/api/overview").json()
        assert ov["pending_approvals_count"] >= 1
        assert ov["failed_executions_count"] >= 1

        # 3. /api/pending-interrupts feeds the /approvals page.
        pending = client.get("/api/pending-interrupts").json()
        rows = [r for r in pending["items"] if r["execution_id"] == execution_id]
        assert len(rows) == 1
        row = rows[0]
        assert row["chain_name"] == "ui-durability-chain"
        assert row["reason"] == "manager_approval"
        assert row["context"] == {"amount": 50_000, "policy": "high-value"}

        # 4. /api/executions/<id> feeds the /executions/:id page.
        exec_view = client.get(f"/api/executions/{execution_id}").json()
        assert exec_view["status"] == "interrupted"
        assert exec_view["chain_name"] == "ui-durability-chain"
        assert exec_view["checkpoint_count"] >= 1
        node_ids = {cp["node_id"] for cp in exec_view["checkpoints"]}
        assert "approval" in node_ids

        # 5. The Approve button fires this exact request.
        resume = client.post(
            f"/api/executions/{execution_id}/resume",
            json={
                "approved": True,
                "metadata": {"approver": "alice"},
                "reason": "ok",
            },
        )
        assert resume.status_code == 200, resume.text
        body = resume.json()
        assert body["result"]["status"] == "completed"

        # 6. After Approve, the row is gone and the counters drop.
        pending_after = client.get("/api/pending-interrupts").json()
        assert all(r["execution_id"] != execution_id for r in pending_after["items"]), pending_after
        ov_after = client.get("/api/overview").json()
        assert ov_after["pending_approvals_count"] < ov["pending_approvals_count"]

        # And the execution view now shows completed nodes for both steps.
        exec_after = client.get(f"/api/executions/{execution_id}").json()
        # The final node must have run and completed; the resume re-executed
        # the suspended approval node first.
        node_ids_after = [cp["node_id"] for cp in exec_after["checkpoints"]]
        assert "final" in node_ids_after, node_ids_after
        # Latest is now ``completed``, not ``interrupted``.
        assert exec_after["status"] == "completed"

    def test_double_approve_returns_409(self, tmp_path: Path) -> None:
        """A double-clicked Approve button: the second hit gets 409 Conflict."""
        require_env()

        db_path = str(tmp_path / "ui-gate-409.db")
        chain = _build_chain(db_path)
        execution_id = f"ui-409-{uuid.uuid4().hex[:8]}"
        chain.execute({"amount": 50_000}, execution_id=execution_id)

        app = build_app(db_path=db_path, no_auth=True, runners=[chain])
        client = TestClient(app)

        first = client.post(
            f"/api/executions/{execution_id}/resume",
            json={"approved": True, "metadata": {}},
        )
        assert first.status_code == 200, first.text

        second = client.post(
            f"/api/executions/{execution_id}/resume",
            json={"approved": True, "metadata": {}},
        )
        assert second.status_code == 409, second.text
        assert "already resumed" in second.json()["detail"].lower()
