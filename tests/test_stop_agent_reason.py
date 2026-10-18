"""A ``StopAgent`` ends the run, keeps its reason, and is not re-driven (1.90.0, F8).

``StopAgent(message, reason=...)`` is middleware's cooperative stop. The loop
turned it into plain content and returned, and two things went wrong after:

* the ``reason`` was dropped — nothing on the result said the run had stopped;
* with ``output_type`` set, the stop message failed to parse, so the structured
  re-ask called the model again (outside middleware, so a ``ToolBudget`` or a
  cost stopper never saw it) and the model's answer **replaced** the stop. A
  ``reask`` output guardrail could do the same.

And ``astream`` dropped the stop message: a streaming caller saw only
``StreamDone``.

Since 1.90.0 ``AgentResult.stop_reason`` carries the reason (``"stopped"`` when
none was given), no re-ask follows a stop (output guardrails still judge the
stop message), and ``astream`` yields the message as one ``TextDelta``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from fastaiagent import Agent, FunctionTool
from fastaiagent._internal.errors import GuardrailBlockedError, StopAgent
from fastaiagent.agent.middleware import AgentMiddleware
from fastaiagent.guardrail.guardrail import (
    Guardrail,
    GuardrailPosition,
    GuardrailResult,
    GuardrailType,
)
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.llm.stream import StreamDone, TextDelta
from fastaiagent.testing import FunctionModel


class Out(BaseModel):
    answer: str


class _Counting:
    """Asks for one tool call, then answers with valid JSON; counts the calls."""

    def __init__(self) -> None:
        self.calls = 0

    def respond(self, messages: list[Any]) -> LLMResponse:
        self.calls += 1
        if not any(m.role.value == "tool" for m in messages):
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="lookup", arguments={"q": "x"})],
                finish_reason="tool_calls",
            )
        return LLMResponse(content='{"answer": "from the model"}', finish_reason="stop")


class _Stopper(AgentMiddleware):
    name = "stopper"

    def __init__(self, where: str, reason: str = "budget") -> None:
        self.where, self.reason = where, reason

    def _stop(self) -> None:
        raise StopAgent("budget exhausted", reason=self.reason)

    async def before_model(self, ctx, messages):
        if self.where == "before_model":
            self._stop()
        return messages

    async def after_model(self, ctx, response):
        if self.where == "after_model":
            self._stop()
        return response

    async def wrap_tool(self, ctx, tool, args, call_next):
        if self.where == "wrap_tool":
            self._stop()
        return await call_next(tool, args)


def _agent(where: str, model: _Counting, *, structured: bool, **kw: Any) -> Agent:
    return Agent(
        name="stopper",
        llm=FunctionModel(model.respond),
        tools=[FunctionTool(name="lookup", fn=lambda q: f"found {q}")],
        middleware=[_Stopper(where, **kw)],
        output_type=Out if structured else None,
    )


WHERE = ["before_model", "after_model", "wrap_tool"]
#: How many model calls each stop point lets through before it fires.
CALLS_BEFORE_STOP = {"before_model": 0, "after_model": 1, "wrap_tool": 1}


@pytest.mark.parametrize("structured", [False, True], ids=["text", "output_type"])
@pytest.mark.parametrize("where", WHERE)
def test_arun_keeps_the_stop_and_its_reason(where: str, structured: bool) -> None:
    model = _Counting()
    result = _agent(where, model, structured=structured).run("go", trace=False)

    assert result.output == "budget exhausted"
    assert result.stop_reason == "budget"
    assert result.parsed is None
    assert model.calls == CALLS_BEFORE_STOP[where], "the stopped run called the model again"


def test_a_stop_without_a_reason_still_says_it_stopped() -> None:
    model = _Counting()
    result = _agent("before_model", model, structured=False, reason="").run("go", trace=False)
    assert result.stop_reason == "stopped"


def test_a_run_that_was_not_stopped_has_no_stop_reason() -> None:
    model = _Counting()
    agent = Agent(
        name="free",
        llm=FunctionModel(model.respond),
        tools=[FunctionTool(name="lookup", fn=lambda q: f"found {q}")],
        output_type=Out,
    )
    result = agent.run("go", trace=False)
    assert result.stop_reason is None
    assert result.parsed == Out(answer="from the model")


def test_a_reask_output_guardrail_does_not_re_drive_a_stopped_run() -> None:
    """It still judges the stop message; with no re-ask allowed, a failing
    ``reask`` rule blocks rather than asking the model for a new answer."""
    model = _Counting()
    rule = Guardrail(
        name="must-say-please",
        guardrail_type=GuardrailType.code,
        position=GuardrailPosition.output,
        fn=lambda text: GuardrailResult(passed="please" in text, message="say please"),
        action="reask",
    )
    agent = Agent(
        name="stopper",
        llm=FunctionModel(model.respond),
        middleware=[_Stopper("before_model")],
        guardrails=[rule],
    )
    with pytest.raises(GuardrailBlockedError):
        agent.run("go", trace=False)
    assert model.calls == 0, "a reask guardrail re-drove a stopped run"


@pytest.mark.parametrize("where", WHERE)
def test_astream_yields_the_stop_message(where: str) -> None:
    model = _Counting()
    agent = _agent(where, model, structured=False)

    async def drain() -> list[Any]:
        return [e async for e in agent.astream("go", trace=False)]

    events = asyncio.run(drain())
    texts = [e.text for e in events if isinstance(e, TextDelta)]
    assert texts and texts[-1] == "budget exhausted", events
    assert isinstance(events[-1], StreamDone)
    assert model.calls == CALLS_BEFORE_STOP[where]
