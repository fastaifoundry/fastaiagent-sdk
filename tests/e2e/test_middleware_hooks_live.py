"""The run-level middleware hooks against real models (1.91.0, Bet 1 piece 1A).

Offline, ``tests/test_middleware_run_hooks.py`` drives every hook with a recording
model; this proves the parts a real provider can break: ``wrap_model`` on a real
token stream still delivers the deltas as they arrive, ``ctx.llm`` sends one call
to a different real model, and ``before_run`` / ``after_run`` bracket a real tool
loop exactly once.

Run with the keys from ``~/.zshrc``::

    zsh -lc 'unset FASTAIAGENT_API_KEY FASTAIAGENT_TARGET; \
      FASTAIAGENT_LOCAL_DB=<scratch>/local.db \
      .venv/bin/python -m pytest tests/e2e/test_middleware_hooks_live.py -m e2e -v -rs'
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

pytestmark = pytest.mark.e2e

MODELS = [
    pytest.param("openai", "gpt-5-mini", "OPENAI_API_KEY", id="gpt-5-mini"),
    pytest.param("gemini", "gemini-3.8-flash", "GEMINI_API_KEY", id="gemini-3.8-flash"),
    pytest.param("anthropic", "claude-haiku-4-5", "ANTHROPIC_API_KEY", id="claude-haiku-4-5"),
]


def _need(key: str) -> None:
    if not os.environ.get(key):
        pytest.skip(f"{key} not set")


def get_order_status(order_id: str) -> str:
    """Status of an order."""
    return {"A-17": "shipped on 2026-10-14"}.get(order_id, "unknown order")


def _agent(provider: str, model: str, *middleware):
    from fastaiagent import Agent, FunctionTool, LLMClient

    return Agent(
        name="hooks-live",
        system_prompt="Use the order tool, then answer in one or two sentences.",
        llm=LLMClient(provider=provider, model=model),
        tools=[FunctionTool(name="get_order_status", fn=get_order_status)],
        middleware=list(middleware),
    )


@pytest.mark.parametrize(("provider", "model", "key"), MODELS)
def test_hooks_bracket_a_real_tool_loop(provider: str, model: str, key: str) -> None:
    _need(key)
    from fastaiagent import AgentMiddleware

    log: list[str] = []

    class Trace(AgentMiddleware):
        async def before_run(self, ctx, input):
            log.append("before_run")
            return f"{input} (customer tier: gold)"

        async def wrap_model(self, ctx, messages, call_next):
            log.append(f"model:{ctx.turn}")
            return await call_next(messages)

        async def after_run(self, ctx, result):
            log.append("after_run")
            result.output = result.output.strip()
            return result

    result = _agent(provider, model, Trace()).run("Where is order A-17?")
    assert result.status == "completed"
    assert "2026-10-14" in result.output or "shipped" in result.output.lower()
    assert log[0] == "before_run" and log[-1] == "after_run"
    assert log.count("before_run") == 1 and log.count("after_run") == 1
    assert [e for e in log if e.startswith("model:")][:2] == ["model:0", "model:1"]


@pytest.mark.parametrize(("provider", "model", "key"), MODELS)
def test_wrap_model_on_astream_still_streams_live(provider: str, model: str, key: str) -> None:
    """The deltas must reach the caller while the model is still answering — not
    all at once after ``wrap_model`` returns."""
    _need(key)
    from fastaiagent import AgentMiddleware
    from fastaiagent.llm.stream import StreamDone, TextDelta

    returned_at: list[float] = []

    class Timed(AgentMiddleware):
        async def wrap_model(self, ctx, messages, call_next):
            response = await call_next(messages)
            returned_at.append(time.monotonic())
            return response

    async def drain():
        agent = _agent(provider, model, Timed())
        agent.system_prompt = "Answer in about 120 words. Do not use tools."
        stamps: list[float] = []
        events = []
        async for event in agent.astream("Explain why order tracking matters to customers."):
            events.append(event)
            if isinstance(event, TextDelta):
                stamps.append(time.monotonic())
        return events, stamps

    events, stamps = asyncio.run(drain())
    assert isinstance(events[-1], StreamDone)
    assert len(stamps) > 3, "the reply arrived in too few deltas to tell"
    # The first delta arrived before the (last) model call returned through the onion.
    assert stamps[0] < returned_at[-1]


def test_ctx_llm_sends_one_call_to_another_real_model() -> None:
    _need("OPENAI_API_KEY")
    _need("GEMINI_API_KEY")
    from fastaiagent import AgentMiddleware, LLMClient

    cheap = LLMClient(provider="gemini", model="gemini-3.5-flash-lite")
    models: list[str] = []

    class FinalAnswerOnCheapModel(AgentMiddleware):
        async def before_model(self, ctx, messages):
            if ctx.turn >= 1:
                ctx.llm = cheap
            return messages

        async def wrap_model(self, ctx, messages, call_next):
            models.append(ctx.llm.model)
            return await call_next(messages)

    agent = _agent("openai", "gpt-5-mini", FinalAnswerOnCheapModel())
    result = agent.run("Where is order A-17?")
    assert result.status == "completed"
    assert models[0] == "gpt-5-mini"
    assert models[1:] and set(models[1:]) == {"gemini-3.5-flash-lite"}
    assert agent.llm.model == "gpt-5-mini"
