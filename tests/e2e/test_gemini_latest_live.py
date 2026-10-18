"""Tool-using agents on Gemini 3.5 and later, against the real API (1.91.0, X11).

Every Gemini model after 2.5 signs the first function call of a turn and rejects
the next request with HTTP 400 when the signature is not sent back. Before 1.91.0
the SDK dropped it, so an agent with tools failed on its second turn on every
current Gemini model — ``run`` and ``astream`` alike. This runs the real tool loop
on the latest models, a resume from a checkpoint (the signature must survive it),
and a history the model never signed (another provider's), and checks each run
is priced.

Run with the keys from ``~/.zshrc``::

    zsh -lc 'unset FASTAIAGENT_API_KEY FASTAIAGENT_TARGET; \
      FASTAIAGENT_LOCAL_DB=<scratch>/local.db \
      .venv/bin/python -m pytest tests/e2e/test_gemini_latest_live.py -m e2e -v -rs'
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.e2e

#: (model, priced?) — the 3.5+ range, plus the alias the UI suggests first.
LATEST = [
    pytest.param("gemini-3.8-flash", True, id="gemini-3.8-flash"),
    pytest.param("gemini-4-argon", True, id="gemini-4-argon"),
    pytest.param("gemini-3.5-flash-lite", True, id="gemini-3.5-flash-lite"),
    pytest.param("gemini-flash-latest", False, id="gemini-flash-latest"),
]

SYSTEM = (
    "Use the tools. First call get_temperature for Oslo; after you see that result, call "
    "get_population for Tokyo. Then answer in one line as 'T=<temp>; P=<pop>'."
)


def _need() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")


def get_temperature(city: str) -> str:
    """Current temperature in a city."""
    return {"oslo": "-4C", "tokyo": "23C"}.get(city.lower(), "unknown")


def get_population(city: str) -> str:
    """Population of a city."""
    return {"oslo": "0.7 million", "tokyo": "14 million"}.get(city.lower(), "unknown")


def _agent(model: str, **kw):
    from fastaiagent import Agent, FunctionTool, LLMClient

    return Agent(
        name="gemini-latest",
        system_prompt=kw.pop("system_prompt", SYSTEM),
        llm=LLMClient(provider="gemini", model=model),
        tools=kw.pop(
            "tools",
            [
                FunctionTool(name="get_temperature", fn=get_temperature),
                FunctionTool(name="get_population", fn=get_population),
            ],
        ),
        **kw,
    )


@pytest.mark.parametrize(("model", "priced"), LATEST)
def test_the_tool_loop_runs_on_run(model: str, priced: bool) -> None:
    _need()
    result = _agent(model).run("Go.")
    assert result.status == "completed"
    assert len(result.tool_calls) >= 2
    assert "-4" in result.output and "14" in result.output
    assert result.cost_known is priced
    if priced:
        assert result.cost > 0


@pytest.mark.parametrize(("model", "priced"), LATEST)
def test_the_tool_loop_runs_on_astream(model: str, priced: bool) -> None:
    _need()
    from fastaiagent.llm.stream import StreamDone, TextDelta, ToolCallEnd

    async def drain():
        return [e async for e in _agent(model).astream("Go.")]

    events = asyncio.run(drain())
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert sum(isinstance(e, ToolCallEnd) for e in events) >= 2
    assert isinstance(events[-1], StreamDone)
    assert "-4" in text and "14" in text


def test_a_resumed_run_keeps_its_signature(tmp_path) -> None:
    """The pause is checkpointed with the signed call; the request after the
    resume must still carry it or Gemini 3 rejects it."""
    _need()
    from fastaiagent import FunctionTool, Resume, SQLiteCheckpointer, interrupt

    def refund(order: str) -> str:
        """Refund an order (needs approval)."""
        interrupt(reason="approve refund?", context={"order": order})
        return f"refunded {order}"

    agent = _agent(
        "gemini-3.8-flash",
        system_prompt="Call refund for the order the user names, then confirm in one line.",
        tools=[FunctionTool(name="refund", fn=refund)],
        checkpointer=SQLiteCheckpointer(db_path=str(tmp_path / "cp.db")),
    )
    paused = agent.run("Refund order A-17.", execution_id="gemini-latest-resume")
    assert paused.status == "paused"
    done = agent.resume("gemini-latest-resume", resume_value=Resume(approved=True))
    assert done.status == "completed"
    assert "A-17" in done.output or "refund" in done.output.lower()


def test_another_providers_history_is_accepted() -> None:
    """A tool turn the Gemini model never signed — here written by hand, as any
    other provider or ``messages=`` would leave it — must not 400."""
    _need()
    from fastaiagent.llm.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

    history = [
        UserMessage("Temperature of Oslo and population of Tokyo?"),
        AssistantMessage(
            "",
            tool_calls=[
                ToolCall(id="call_a", name="get_temperature", arguments={"city": "Oslo"}),
                ToolCall(id="call_b", name="get_population", arguments={"city": "Tokyo"}),
            ],
        ),
        ToolMessage("-4C", tool_call_id="call_a"),
        ToolMessage("14 million", tool_call_id="call_b"),
    ]
    result = _agent("gemini-4-argon", system_prompt="Answer as 'T=<temp>; P=<pop>'.").run(
        "Answer now.", messages=history
    )
    assert "-4" in result.output and "14" in result.output
