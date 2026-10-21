"""1.90.0 "Fix now" against real models (OpenAI GPT-5-class, Anthropic Claude
Haiku 4.5, Gemini).

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
        llm=LLM(model="gpt-5-mini"),
    )
    task = Task(
        description="What colour is the sky about {topic}?",
        expected_output="one line",
        agent=analyst,
    )
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


# ---------------------------------------------------------------------------
# F6: AgentConfig.tool_choice / temperature reach real providers
# ---------------------------------------------------------------------------

CHOICE_MODELS = [
    *MODELS,
    pytest.param("gemini", "gemini-2.5-flash", "GEMINI_API_KEY", id="gemini-2.5-flash"),
]


def _lookup_agent(provider: str, model: str, **config):
    from fastaiagent import Agent, AgentConfig, FunctionTool, LLMClient

    calls: list[str] = []

    def lookup_order(order: str) -> str:
        """Look up the shipping status of an order by its number."""
        calls.append(order)
        return f"order {order} shipped yesterday"

    agent = Agent(
        name=f"choice-{model}",
        system_prompt="You are a helpful assistant.",
        llm=LLMClient(provider=provider, model=model),
        tools=[FunctionTool(name="lookup_order", fn=lookup_order)],
        config=AgentConfig(**config),
    )
    return agent, calls


@pytest.mark.parametrize(("provider", "model", "key"), CHOICE_MODELS)
def test_tool_choice_required_forces_the_first_turn_on_a_real_model(provider, model, key):
    _need(key)
    agent, calls = _lookup_agent(provider, model, tool_choice="required")
    # A prompt no model needs a tool for: only tool_choice makes it call one.
    result = agent.run("Say hello in one word. Order 77 exists, by the way.")
    assert calls, f"{model} answered without the forced tool call"
    assert result.output.strip(), f"{model} never answered after the forced turn"


@pytest.mark.parametrize(("provider", "model", "key"), CHOICE_MODELS)
def test_tool_choice_none_keeps_a_real_model_off_its_tools(provider, model, key):
    _need(key)
    agent, calls = _lookup_agent(provider, model, tool_choice="none")
    # Not a prompt that DEMANDS the tool: told to use one under tool_choice="none",
    # Claude returns an empty content list and Gemini ends with
    # UNEXPECTED_TOOL_CALL — no text from either (seen live, 2026-10-18; the
    # providers' behaviour, documented in docs/agents/index.md).
    result = agent.run("What is 2 + 2? Order 77 is in the system, if that matters.")
    assert calls == [], f"{model} called a tool with tool_choice='none'"
    assert "4" in result.output, f"{model}: {result.output!r}"


def test_agent_temperature_is_sent_and_honoured_by_a_model_that_accepts_it():
    # Stays on gpt-4o-mini on purpose: the point is a model that accepts a
    # non-default temperature (gpt-5-mini rejects one — the next test).
    _need("OPENAI_API_KEY")
    agent, _ = _lookup_agent("openai", "gpt-4o-mini", temperature=0.0, max_tokens=40)
    assert agent.run("Reply with the single word: ready").output.strip()


def test_a_reasoning_model_rejects_a_non_default_temperature_itself():
    """1.89.0 decision D4, unchanged: nothing is dropped for a model's sampling
    rules. AgentConfig.temperature used to be ignored; now it is sent, and a model
    that only accepts the default says so — the provider is the authority."""
    _need("OPENAI_API_KEY")
    from fastaiagent._internal.errors import LLMProviderError

    agent, _ = _lookup_agent("openai", "gpt-5-mini", temperature=0.2)
    with pytest.raises(LLMProviderError, match="(?i)temperature"):
        agent.run("Reply with the single word: ready")


# ---------------------------------------------------------------------------
# F7: a real Gemini thinking model's tokens add up, and are priced
# ---------------------------------------------------------------------------


def test_gemini_thinking_tokens_are_completion_tokens():
    _need("GEMINI_API_KEY")
    from fastaiagent import Agent, LLMClient
    from fastaiagent.llm.message import UserMessage

    llm = LLMClient(provider="gemini", model="gemini-2.5-flash")
    question = "A bat and a ball cost $1.10; the bat costs $1 more than the ball. The ball?"
    resp = asyncio.run(llm.acomplete([UserMessage(question)]))
    usage = resp.usage
    assert usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"], usage

    result = Agent(name="gemini-think", llm=llm).run(question)
    assert result.cost_known and result.cost > 0


# ---------------------------------------------------------------------------
# F15 consequence: the [anthropic] extra widened to anthropic 1.x
# ---------------------------------------------------------------------------


def test_the_anthropic_integration_traces_a_real_call_on_the_installed_sdk(tmp_path):
    """Pydantic AI 2.x needs anthropic>=1.12.1, so moving [pydanticai] to 2.x
    widened [anthropic] past <1. ``integrations.anthropic`` patches the SDK's
    ``messages.create``; this drives it against the real API on whichever
    ``anthropic`` is installed (run on 0.105 and 1.15 for 1.90.0)."""
    _need("ANTHROPIC_API_KEY")
    anthropic = pytest.importorskip("anthropic")
    import sqlite3

    from fastaiagent.integrations import anthropic as fa_anthropic
    from fastaiagent.trace.otel import get_tracer_provider

    fa_anthropic.enable()
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=20,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        )
        assert msg.content and msg.content[0].text.strip()
    finally:
        fa_anthropic.disable()

    provider = get_tracer_provider()
    if provider is not None and hasattr(provider, "force_flush"):
        provider.force_flush()
    db = os.environ.get("FASTAIAGENT_LOCAL_DB")
    if not db:
        pytest.skip("set FASTAIAGENT_LOCAL_DB to read the captured span")
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT attributes FROM spans WHERE name LIKE '%anthropic%' ORDER BY start_time DESC"
        ).fetchall()
    finally:
        conn.close()
    assert rows, f"no anthropic span captured on anthropic {anthropic.__version__}"
    import json as _json

    attrs = _json.loads(rows[0][0] or "{}")
    assert int(attrs.get("gen_ai.usage.input_tokens") or 0) > 0, attrs


# ---------------------------------------------------------------------------
# F20: the [openai] extra widened to openai 3.x
# ---------------------------------------------------------------------------


def _span_attrs(db: str, name_like: str) -> list[dict]:
    import json as _json
    import sqlite3

    from fastaiagent.trace.otel import get_tracer_provider

    provider = get_tracer_provider()
    if provider is not None and hasattr(provider, "force_flush"):
        provider.force_flush()
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT attributes FROM spans WHERE name LIKE ? ORDER BY start_time DESC",
            (name_like,),
        ).fetchall()
    finally:
        conn.close()
    return [_json.loads(r[0] or "{}") for r in rows]


@pytest.mark.parametrize("stream", [False, True], ids=["create", "stream"])
def test_the_openai_integration_traces_a_real_call_on_the_installed_sdk(stream):
    """``integrations.openai`` patches the SDK's chat-completions resource; this
    drives it against the real API on whichever ``openai`` is installed (run on
    2.54 and 3.35 for 1.90.0)."""
    _need("OPENAI_API_KEY")
    openai = pytest.importorskip("openai")
    from fastaiagent.integrations import openai as fa_openai

    db = os.environ.get("FASTAIAGENT_LOCAL_DB")
    if not db:
        pytest.skip("set FASTAIAGENT_LOCAL_DB to read the captured span")
    fa_openai.enable()
    try:
        client = openai.OpenAI()
        kwargs = {
            # GPT-5 takes max_completion_tokens (not max_tokens); minimal
            # reasoning keeps the budget for the reply.
            "model": "gpt-5-mini",
            "reasoning_effort": "minimal",
            "max_completion_tokens": 64,
            "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        }
        if stream:
            text = "".join(
                (c.choices[0].delta.content or "") if c.choices else ""
                for c in client.chat.completions.create(
                    stream=True, stream_options={"include_usage": True}, **kwargs
                )
            )
        else:
            text = client.chat.completions.create(**kwargs).choices[0].message.content
        assert (text or "").strip()
    finally:
        fa_openai.disable()

    spans = _span_attrs(db, "openai.chat.%")
    assert spans, f"no openai span captured on openai {openai.__version__}"
    assert int(spans[0].get("gen_ai.usage.input_tokens") or 0) > 0, spans[0]


@pytest.mark.parametrize("mode", ["run", "astream"])
def test_an_agent_on_an_injected_openai_client_runs_on_the_installed_sdk(mode):
    """``LLMClient(openai_client=...)`` delegates every call to the SDK client."""
    _need("OPENAI_API_KEY")
    openai = pytest.importorskip("openai")
    from fastaiagent import Agent, FunctionTool, LLMClient

    def lookup_order(order: str) -> str:
        """Look up the shipping status of an order by its number."""
        return f"order {order} shipped yesterday"

    agent = Agent(
        name="injected-client",
        system_prompt="Use lookup_order to answer order questions.",
        llm=LLMClient(
            provider="openai",
            model="gpt-5-mini",
            reasoning="minimal",
            openai_client=openai.AsyncOpenAI(),
        ),
        tools=[FunctionTool(name="lookup_order", fn=lookup_order)],
    )
    if mode == "run":
        result = agent.run("Where is order 77?")
        assert "shipped" in result.output.lower(), result.output
        assert result.tool_calls, "the injected client never made the tool call"
    else:
        from fastaiagent.llm.stream import TextDelta

        async def drain() -> str:
            return "".join(
                [
                    e.text
                    async for e in agent.astream("Where is order 77?")
                    if isinstance(e, TextDelta)
                ]
            )

        assert "shipped" in asyncio.run(drain()).lower()


def test_the_openai_embedder_and_moderation_work_on_the_installed_sdk():
    """The other two places the SDK calls the openai package: KB embeddings and
    the moderation-backed safety detector."""
    _need("OPENAI_API_KEY")
    pytest.importorskip("openai")
    from fastaiagent._internal.safety_detectors import moderate_text
    from fastaiagent.kb.embedding import OpenAIEmbedder

    vectors = OpenAIEmbedder().embed(["refund order 1042", "shipping status"])
    assert len(vectors) == 2 and len(vectors[0]) > 100

    result = moderate_text("I would like to bake a cake for my friend's birthday.")
    assert result.flagged is False
