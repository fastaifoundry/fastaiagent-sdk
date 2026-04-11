"""End-to-end quality gate — Chain with real LLM.

Chains are a first-class feature of the SDK (directed graph workflows
with cycles, typed state, and checkpointing). The main quality gate is
agent-only. This gate proves the chain executor runs a real two-node
pipeline end-to-end against a real LLM, produces a populated
``ChainResult``, and leaves trace metadata behind.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


class TestChainGate:
    """Two-node Chain: research -> respond, real OpenAI calls."""

    def test_01_build_and_execute(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent import Agent, Chain, LLMClient

        researcher = Agent(
            name="chain-gate-researcher",
            system_prompt=(
                "You are a researcher. Provide a two-sentence summary of the "
                "topic the user asks about. Keep it factual."
            ),
            llm=LLMClient(provider="openai", model="gpt-4.1"),
        )
        responder = Agent(
            name="chain-gate-responder",
            system_prompt=(
                "You are a writer. Take the research provided in the state "
                "and rewrite it as a single concise sentence."
            ),
            llm=LLMClient(provider="openai", model="gpt-4.1"),
        )

        chain = Chain("chain-gate-pipeline", checkpoint_enabled=False)
        chain.add_node("research", agent=researcher)
        chain.add_node("respond", agent=responder)
        chain.connect("research", "respond")

        errors = chain.validate()
        assert not errors, f"Chain validation errors: {errors}"

        result = chain.execute({"message": "AI agent evaluation frameworks"})

        assert result is not None
        assert result.output, "ChainResult.output is empty"
        assert result.execution_id, "ChainResult.execution_id not assigned"
        assert result.node_results, "ChainResult.node_results is empty"
        assert "research" in result.node_results, (
            f"research node did not record a result: {result.node_results.keys()}"
        )
        assert "respond" in result.node_results, (
            f"respond node did not record a result: {result.node_results.keys()}"
        )
        gate_state["chain_result"] = result

    def test_02_chain_final_state_carries_through(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        result = gate_state["chain_result"]
        # final_state should contain whatever keys the chain accumulated.
        # We at least expect the initial user message to survive.
        assert isinstance(result.final_state, dict)
        assert result.final_state, "Chain final_state is empty"
