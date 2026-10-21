"""Live replay isolation (1.89.0): a rerun against a real model stays an experiment.

A real gpt-5-mini agent charges a card and looks up an order. Its live rerun,
with a changed prompt, must:

* not charge the card again (blocked by default), and say so;
* run the read-only lookup when allowed by name;
* write a trace marked ``fastaiagent.source=replay`` that is never exported
  (rows land ``synced=1``) and that production readers leave out.

The offline sweep (``tests/integration/test_replay_isolation_sweep.py``) covers
every surface and mode; this gate proves it holds against a real provider,
where the rerun's tool calls are the model's own choice.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e

SYSTEM = (
    "You are a billing agent. For every request: call lookup_order with the order id, "
    "then call charge_card with the order id and the amount from the lookup. "
    "Then confirm in one sentence."
)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    monkeypatch.delenv("FASTAIAGENT_TRACE_DB_PATH", raising=False)
    reset_config()
    otel.reset()
    yield db
    otel.reset()
    reset_config()


def test_a_live_rerun_does_not_repeat_side_effects(world: Path) -> None:
    require_env()
    from fastaiagent import Agent, FunctionTool, LLMClient
    from fastaiagent.trace import TraceStore, otel
    from fastaiagent.trace.replay import Replay

    charges: list[str] = []
    lookups: list[str] = []

    def lookup_order(order_id: str) -> str:
        """Look up an order: returns its amount due."""
        lookups.append(order_id)
        return f"Order {order_id}: amount due 42.00"

    def charge_card(order_id: str, amount: float) -> str:
        """Charge the customer's card for an order."""
        charges.append(order_id)
        return f"charged {amount:.2f} for {order_id}"

    agent = Agent(
        name="live-billing",
        system_prompt=SYSTEM,
        llm=LLMClient(provider="openai", model="gpt-5-mini", reasoning="minimal"),
        tools=[
            FunctionTool(name="lookup_order", fn=lookup_order, replay_class="read_only"),
            FunctionTool(name="charge_card", fn=charge_card),
        ],
    )
    result = agent.run("Please bill order ORD-77.")
    otel.get_tracer_provider().force_flush()
    assert charges == ["ORD-77"], f"the original run did not charge once: {charges}"
    lookups_before = len(lookups)

    rerun = (
        Replay.load(result.trace_id)
        .fork_at(step=0)
        .modify_prompt(SYSTEM + " Be brief.")
        .with_tool_policy("block", allow=["lookup_order"])
        .rerun()
    )
    otel.get_tracer_provider().force_flush()

    assert charges == ["ORD-77"], "the live rerun charged the card again"
    assert rerun.tool_dispositions == {"lookup_order": "live", "charge_card": "blocked"}
    assert len(lookups) > lookups_before, "the allowed read-only tool did not run"
    # The model chose to call charge_card again; the call was blocked, and counted.
    assert rerun.blocked_tool_calls.get("charge_card", 0) >= 1, rerun.blocked_tool_calls
    assert rerun.exported is False
    assert rerun.source_trace_id == result.trace_id

    root = TraceStore().get_trace(rerun.trace_id).spans[0]
    assert root.attributes.get("fastaiagent.source") == "replay"
    assert root.attributes.get("fastaiagent.replay.source_trace_id") == result.trace_id

    with sqlite3.connect(world) as conn:
        synced = {
            row[0]
            for row in conn.execute(
                "SELECT synced FROM spans WHERE trace_id = ?", (rerun.trace_id,)
            )
        }
    assert synced == {1}, f"rerun rows would be exported: synced={synced}"

    production = {t.trace_id for t in TraceStore().list_traces(exclude_sources=True)}
    assert result.trace_id in production
    assert rerun.trace_id not in production
