"""End-to-end quality gate — Anthropic (Claude) provider variant.

Mirrors ``tests/e2e/test_quality_gate.py`` but swaps the LLM provider to
Anthropic so any provider-specific drift in the agent/trace/replay flow
surfaces with the same loud failure mode that caught the missing
``llm.*`` span bug on the OpenAI path.

Scope of assertions is deliberately narrower than the OpenAI gate — this
file trusts that the OpenAI gate covers platform push and eval. Anthropic
exercises agent run, Phase A instrumentation, tool spans, load replay,
fork, rerun, compare.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.e2e.conftest import require_anthropic, require_env

pytestmark = pytest.mark.e2e


def _lookup_order(order_id: str) -> str:
    """Look up an order by ID."""
    orders = {
        "ORD-101": "Acme Widget, shipped 2026-04-02, delivered 2026-04-05",
        "ORD-102": "Spacely Sprocket, processing",
        "ORD-103": "Cogswell Cog, cancelled",
    }
    return orders.get(order_id, f"Order {order_id} not found.")


class TestQualityGateAnthropic:
    """Anthropic-provider quality gate — agent run through replay/compare."""

    def test_01_create_anthropic_agent(self, gate_state: dict[str, Any]) -> None:
        require_env()
        require_anthropic()
        from fastaiagent import Agent, FunctionTool, LLMClient

        agent = Agent(
            name="anthropic-gate-support",
            system_prompt=(
                "You are a customer support agent for Acme Corp. "
                "Use the lookup_order tool to check order status when asked. "
                "Be concise."
            ),
            llm=LLMClient(provider="anthropic", model="claude-sonnet-4-20250514"),
            tools=[FunctionTool(name="lookup_order", fn=_lookup_order)],
        )
        assert agent.name == "anthropic-gate-support"
        assert agent.llm.provider == "anthropic"
        assert len(agent.tools) == 1
        gate_state["anthropic_agent"] = agent

    def test_02_run_agent(self, gate_state: dict[str, Any]) -> None:
        require_env()
        require_anthropic()
        agent = gate_state["anthropic_agent"]
        result = agent.run("What is the status of order ORD-101?")
        assert result.output, "Anthropic agent.run returned empty output"
        lower = result.output.lower()
        assert (
            "ord-101" in lower
            or "shipped" in lower
            or "widget" in lower
            or "delivered" in lower
        ), f"Anthropic LLM did not engage with lookup_order: {result.output!r}"
        assert result.tokens_used > 0, "Anthropic token accounting broken"
        assert result.latency_ms > 0
        assert result.trace_id, "Anthropic run produced no trace_id"
        assert len(result.tool_calls) >= 1, (
            "Anthropic path did not invoke lookup_order — tool-calling broken"
        )
        gate_state["anthropic_result"] = result

    def test_03_load_replay(self, gate_state: dict[str, Any]) -> None:
        require_env()
        require_anthropic()
        from fastaiagent.trace.replay import Replay

        trace_id = gate_state["anthropic_result"].trace_id
        replay = Replay.load(trace_id)
        steps = replay.steps()
        assert len(steps) >= 3, (
            f"expected >=3 spans (root agent + LLM + tool), got {len(steps)}: "
            f"{[s.span_name for s in steps]}"
        )
        root_attrs = replay._trace.spans[0].attributes
        for key in ("agent.config", "agent.tools", "agent.llm.config"):
            assert key in root_attrs, (
                f"Phase A reconstruction attr missing on Anthropic trace: {key}"
            )
        assert root_attrs["agent.llm.provider"] == "anthropic", (
            "agent.llm.provider did not round-trip as 'anthropic'"
        )
        llm_spans = [s for s in steps if s.span_name.startswith("llm.")]
        assert llm_spans, (
            "No llm.* spans in Anthropic trace — LLMClient.acomplete wrap regression"
        )
        gate_state["anthropic_replay"] = replay

    def test_04_fork_rerun_compare(self, gate_state: dict[str, Any]) -> None:
        require_env()
        require_anthropic()
        from fastaiagent.trace.replay import ForkedReplay

        replay = gate_state["anthropic_replay"]
        fork_point = min(2, len(replay.steps()) - 1)
        forked = replay.fork_at(step=fork_point)
        assert isinstance(forked, ForkedReplay)

        forked.modify_prompt(
            "You are a terse support agent. Reply in one sentence maximum."
        )
        rerun_result = forked.rerun()
        assert rerun_result.new_output is not None, (
            "Anthropic rerun returned new_output=None — "
            "reconstruction failed for non-OpenAI provider"
        )
        assert isinstance(rerun_result.new_output, str)
        assert len(rerun_result.new_output) > 0
        assert rerun_result.trace_id, "Anthropic rerun emitted no new trace_id"

        cmp = forked.compare(rerun_result)
        assert cmp.diverged_at == fork_point
        assert len(cmp.new_steps) >= 1, (
            "compare() produced no new_steps for Anthropic rerun"
        )
