"""1.90.0 "Fix now" against real models (OpenAI GPT-4o-mini and GPT-5-class,
Anthropic Claude Haiku 4.5, Gemini).

The offline suites drive each fix with ``FunctionModel`` and a real socket; this
proves the same behaviour holds when a real provider is on the other end:

* F5 — a resumed tool call is judged by the agent's ``tool_result`` guardrail:
  the real model never sees the masked value, on every provider.
* F4 — ``with_guardrails(crew).akickoff()`` runs the guardrails around a real
  CrewAI crew.

Run with the keys from ``~/.zshrc``::

    zsh -lc 'unset FASTAIAGENT_API_KEY FASTAIAGENT_TARGET; \
      FASTAIAGENT_LOCAL_DB=<scratch>/local.db \
      .venv/bin/python -m pytest tests/e2e/test_fix_now_live.py -m e2e -v -rs'
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytestmark = pytest.mark.e2e

SSN = "123-45-6789"

MODELS = [
    pytest.param("openai", "gpt-4o-mini", "OPENAI_API_KEY", id="gpt-4o-mini"),
    pytest.param("openai", "gpt-5.1", "OPENAI_API_KEY", id="gpt-5.1"),
    pytest.param("openai", "gpt-5-mini", "OPENAI_API_KEY", id="gpt-5-mini"),
    pytest.param("anthropic", "claude-haiku-4-5", "ANTHROPIC_API_KEY", id="claude-haiku-4-5"),
]


def _need(key: str) -> None:
    if not os.environ.get(key):
        pytest.skip(f"{key} not set")


# ---------------------------------------------------------------------------
# F5: the resumed call is guarded, against a real model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("provider", "model", "key"), MODELS)
def test_a_resumed_tool_result_is_masked_before_the_real_model_sees_it(
    provider: str, model: str, key: str, tmp_path
) -> None:
    _need(key)
    from fastaiagent import Agent, FunctionTool, LLMClient, Resume, SQLiteCheckpointer, interrupt
    from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType

    ran: list[str] = []
    seen: list[str] = []

    def refund(order: str) -> str:
        """Refund an order. A manager must approve before it runs."""
        if not interrupt(reason="manager_approval", context={"order": order}).approved:
            return "declined"
        ran.append(order)
        result = f"Refunded order {order}. Card holder SSN {SSN}."
        seen.append(result)
        return result

    mask = Guardrail(
        name="mask-ssn",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.tool_result,
        config={"pattern": r"\b\d{3}-\d{2}-\d{4}\b", "should_match": False},
        action="mask",
    )
    agent = Agent(
        name=f"refund-{model}",
        system_prompt=(
            "You process refunds with the refund tool. After it runs, repeat the tool's "
            "result to the user word for word."
        ),
        llm=LLMClient(provider=provider, model=model),
        tools=[FunctionTool(name="refund", fn=refund)],
        guardrails=[mask],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )
    exec_id = f"live-{uuid.uuid4().hex[:8]}"

    first = agent.run("Please refund order 1042.", execution_id=exec_id)
    assert first.status == "paused", first.output
    result = asyncio.run(agent.aresume(exec_id, resume_value=Resume(approved=True)))

    assert result.status == "completed"
    assert ran == ["1042"], "the approved refund did not run"
    assert seen, "the tool never produced its result"
    assert SSN not in result.output, f"{model} repeated the unmasked SSN: {result.output!r}"
    tool_guardrails = [g for g in result.guardrails if g.position == "tool_result"]
    assert tool_guardrails and tool_guardrails[0].action_taken == "masked"


# ---------------------------------------------------------------------------
# F4: a real CrewAI crew, native async
# ---------------------------------------------------------------------------


def test_akickoff_is_guarded_around_a_real_crew() -> None:
    _need("OPENAI_API_KEY")
    pytest.importorskip("crewai")
    from crewai import LLM, Crew, Task
    from crewai import Agent as CrewAgent

    from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailResult, GuardrailType
    from fastaiagent.integrations import crewai as fa_crewai
    from fastaiagent.integrations._registry import GuardrailBlocked

    analyst = CrewAgent(
        role="analyst",
        goal="Answer in one short sentence.",
        backstory="Terse.",
        llm=LLM(model="gpt-4o-mini"),
    )
    task = Task(description="What colour is the sky about {topic}?", expected_output="one line",
                agent=analyst)
    crew = Crew(agents=[analyst], tasks=[task], name="live-desk")
    no_secret = Guardrail(
        name="no-secret-in",
        guardrail_type=GuardrailType.code,
        position=GuardrailPosition.input,
        fn=lambda t: GuardrailResult(passed="SECRET" not in t, message="secret"),
    )
    guarded = fa_crewai.with_guardrails(crew, input_guardrails=[no_secret])

    with pytest.raises(GuardrailBlocked, match="no-secret-in"):
        asyncio.run(guarded.akickoff(inputs={"topic": "SECRET plans"}))
    out = asyncio.run(guarded.akickoff(inputs={"topic": "noon"}))
    assert out.raw.strip(), "the clean run returned nothing"
