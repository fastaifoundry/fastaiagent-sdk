"""End-to-end quality gate — CrewAI integration.

The ``fastaiagent.integrations.crewai`` integration is currently a
minimal ``enable()`` / ``disable()`` stub that only validates the
crewai package is importable. This gate covers:

1. The integration contract (idempotent enable/disable, validates
   crewai is installed).
2. **Interop** — the most important thing: a real CrewAI Crew can run
   alongside a real fastaiagent Agent in the same Python process
   against the same OpenAI key without conflict, and before/after
   the fastaiagent integration is enabled.

This is the kind of test that catches regressions you'd never catch
in unit tests: say a future fastaiagent release inadvertently monkey
patches something LiteLLM (CrewAI's LLM layer) also touches, the
interop test fails loudly even though both SDKs in isolation would
look fine.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.e2e.conftest import require_env, require_import

pytestmark = pytest.mark.e2e


class TestCrewAIIntegrationContract:
    """Integration stub contract — enable/disable idempotence + import validation."""

    def test_01_enable_is_idempotent(self, gate_state: dict[str, Any]) -> None:
        require_env()
        require_import("crewai")
        from fastaiagent.integrations import crewai as crewai_integration

        crewai_integration.enable()
        crewai_integration.enable()  # second call must not raise

    def test_02_disable_flips_state_back(self, gate_state: dict[str, Any]) -> None:
        require_env()
        require_import("crewai")
        from fastaiagent.integrations import crewai as crewai_integration

        crewai_integration.enable()
        assert crewai_integration._enabled is True
        crewai_integration.disable()
        assert crewai_integration._enabled is False
        # And re-enable must still work after disable.
        crewai_integration.enable()
        assert crewai_integration._enabled is True

    def test_03_enable_validates_crewai_importable(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        require_import("crewai")
        # The enable() path performs `import crewai` and raises ImportError
        # if it fails. Since we're gated on require_import above, the
        # import will succeed here; we just verify enable() completes
        # without raising.
        from fastaiagent.integrations import crewai as crewai_integration

        crewai_integration.enable()


class TestCrewAIInteropGate:
    """Real CrewAI Crew + fastaiagent Agent in the same process.

    The crew is intentionally tiny (one agent, one task, one short goal)
    to keep LLM cost and runtime low. The point is to prove the two
    SDKs coexist, not to stress-test CrewAI itself.
    """

    def test_01_fastaiagent_agent_runs_before_crewai(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        require_import("crewai")
        from fastaiagent import Agent, LLMClient

        agent = Agent(
            name="crewai-interop-fastai-before",
            system_prompt="Reply with exactly the single word: alpha",
            llm=LLMClient(provider="openai", model="gpt-4.1"),
        )
        result = agent.run("Say the word.")
        assert result.output, "fastaiagent run produced empty output"
        assert "alpha" in result.output.lower(), (
            f"fastaiagent did not honor the prompt: {result.output!r}"
        )
        gate_state["fastai_before_trace_id"] = result.trace_id

    def test_02_crewai_crew_runs_in_same_process(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        require_import("crewai")
        from fastaiagent.integrations import crewai as crewai_integration

        # Enable the integration BEFORE running the crew — this is the
        # real use case: you've opted into fastaiagent tracing and you're
        # running CrewAI alongside.
        crewai_integration.enable()

        from crewai import Agent as CrewAgent
        from crewai import Crew, Process, Task

        # Single-agent, single-task Crew. Small goal, small output.
        researcher = CrewAgent(
            role="Factual Researcher",
            goal="Provide a one-sentence factual answer.",
            backstory=(
                "You give short, factual, one-sentence answers. "
                "You never use lists, bullets, or multiple sentences."
            ),
            llm="gpt-4.1",
            verbose=False,
            allow_delegation=False,
        )
        task = Task(
            description=(
                "State in one short sentence what the chemical symbol "
                "for water is."
            ),
            expected_output=(
                "One short factual sentence naming the chemical symbol for water."
            ),
            agent=researcher,
        )
        crew = Crew(
            agents=[researcher],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        crew_output = crew.kickoff()

        # CrewOutput stringifies to the final task's result. Assert the
        # crew produced non-empty text and actually answered the question.
        crew_text = str(crew_output).lower()
        assert crew_text, "Crew.kickoff() returned empty output"
        assert "h2o" in crew_text or "h₂o" in crew_text, (
            f"CrewAI crew did not produce the expected chemical symbol: "
            f"{crew_text!r}"
        )
        gate_state["crewai_output"] = crew_text

    def test_03_fastaiagent_agent_still_runs_after_crewai(
        self, gate_state: dict[str, Any]
    ) -> None:
        """Prove fastaiagent is still functional after a real CrewAI run.

        Catches the class of regression where CrewAI's internals (LiteLLM,
        telemetry, OTel setup, etc.) leave the process in a state that
        breaks fastaiagent tracing or LLM dispatch.
        """
        require_env()
        require_import("crewai")
        from fastaiagent import Agent, LLMClient
        from fastaiagent.trace.replay import Replay

        agent = Agent(
            name="crewai-interop-fastai-after",
            system_prompt="Reply with exactly the single word: omega",
            llm=LLMClient(provider="openai", model="gpt-4.1"),
        )
        result = agent.run("Say the word.")
        assert result.output, "fastaiagent run after CrewAI produced empty output"
        assert "omega" in result.output.lower(), (
            f"fastaiagent did not honor the prompt after CrewAI: "
            f"{result.output!r}"
        )
        assert result.trace_id, "trace_id missing — tracing broke after CrewAI"

        # Phase A span attrs must still round-trip after CrewAI's
        # runtime has left its footprint on the process.
        replay = Replay.load(result.trace_id)
        root_attrs = replay._trace.spans[0].attributes
        assert "agent.config" in root_attrs, (
            "agent.config span attr missing — Phase A instrumentation "
            "was perturbed by CrewAI"
        )
        llm_spans = [s for s in replay.steps() if s.span_name.startswith("llm.")]
        assert llm_spans, (
            "llm.* spans missing after CrewAI ran — LLMClient.acomplete "
            "wrap was perturbed"
        )
