"""End-to-end quality gate — Supervisor / Worker multi-agent delegation.

``Supervisor`` wraps a team of ``Worker`` agents and delegates tasks by
tool-calling each worker. Exercises the delegation path, the per-worker
tool generation, and the synthesis step end-to-end with two real workers.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


class TestSupervisorGate:
    """Two-worker Supervisor with real LLM calls."""

    def test_01_build_team_and_run(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent import Agent, LLMClient, Supervisor, Worker

        llm = LLMClient(provider="openai", model="gpt-4.1")

        researcher = Agent(
            name="supervisor-gate-researcher",
            system_prompt=(
                "You are a research specialist. Given a topic, return 2-3 "
                "factual bullet points. Nothing else."
            ),
            llm=llm,
        )
        writer = Agent(
            name="supervisor-gate-writer",
            system_prompt=(
                "You are a writer. Given research notes, rewrite them as a "
                "single flowing sentence suitable for a tweet."
            ),
            llm=llm,
        )

        supervisor = Supervisor(
            name="supervisor-gate-lead",
            llm=llm,
            workers=[
                Worker(
                    agent=researcher,
                    role="researcher",
                    description="Provides factual bullet points on a topic.",
                ),
                Worker(
                    agent=writer,
                    role="writer",
                    description="Rewrites research as a concise sentence.",
                ),
            ],
            max_delegation_rounds=3,
        )

        result = supervisor.run(
            "Research the topic 'OpenTelemetry semantic conventions for GenAI' "
            "and then rewrite the research as a single tweet-length sentence."
        )

        assert result.output, "Supervisor returned empty output"
        assert result.tokens_used > 0
        # Supervisor should have invoked at least one delegation tool.
        assert result.tool_calls, (
            "Supervisor made no tool calls — delegation path broken"
        )
        tool_names = {tc["tool_name"] for tc in result.tool_calls}
        assert any(n.startswith("delegate_to_") for n in tool_names), (
            f"No delegate_to_* tool calls recorded: {tool_names}"
        )
        gate_state["supervisor_result"] = result

    def test_02_supervisor_delegated_to_both_workers(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        result = gate_state["supervisor_result"]
        tool_names = {tc["tool_name"] for tc in result.tool_calls}
        # Realistic expectation: the supervisor SHOULD call both researcher
        # and writer for a "research and then rewrite" task. If it only
        # calls one, either the LLM got lazy or the system prompt isn't
        # directive enough. Either way, worth flagging.
        assert "delegate_to_researcher" in tool_names, (
            f"Supervisor did not delegate to researcher: {tool_names}"
        )
        assert "delegate_to_writer" in tool_names, (
            f"Supervisor did not delegate to writer: {tool_names}"
        )
