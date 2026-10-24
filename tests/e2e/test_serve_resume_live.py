"""Live: ``fastaiagent agent serve`` pauses a gpt-5-mini run on ``/run`` and
finishes it on ``/resume`` (1.92.0). The offline cases (409, 404, token, Chain)
are in ``tests/test_cli_agent_serve.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent import Agent, FunctionTool, LLMClient, interrupt  # noqa: E402
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer  # noqa: E402
from fastaiagent.cli import agent as agent_cli  # noqa: E402

pytestmark = pytest.mark.e2e


def test_serve_pauses_on_run_and_finishes_on_resume(tmp_path: Path) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    def ask_manager(amount: int) -> str:
        """Ask a manager to approve a refund."""
        decision = interrupt("manager_approval", {"amount": amount})
        return "approved" if decision.approved else "declined"

    agent = Agent(
        name="refunds",
        system_prompt=(
            "For any refund, call ask_manager with the amount first. Then say in one "
            "sentence whether the refund was approved or declined."
        ),
        llm=LLMClient(provider="openai", model="gpt-5-mini", reasoning="minimal"),
        tools=[FunctionTool(name="ask_manager", fn=ask_manager)],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "cp.db")),
    )
    client = TestClient(agent_cli._build_app(agent))

    paused = client.post("/run", json={"input": "Refund 40 dollars for order 1042."}).json()
    assert paused["status"] == "paused", paused
    assert paused["pending_interrupt"]["reason"] == "manager_approval"

    done = client.post(
        "/resume",
        json={"execution_id": paused["execution_id"], "approved": False, "metadata": {}},
    ).json()
    assert done["status"] == "completed", done
    assert "declin" in done["output"].lower()
