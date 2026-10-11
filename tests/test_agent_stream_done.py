"""``Agent.astream()`` ends with exactly one ``StreamDone`` (1.88.0).

The streamed tool loop drops each turn's ``StreamDone`` so a multi-turn run reads
as one stream — and until 1.88.0 nothing emitted the final one, so a consumer
that waited for the end-of-stream marker from an agent waited forever.
``LLMClient.astream()`` always ended with one; now the agent does too, after its
output guardrails and memory write. A paused run still ends with ``Paused``.

Swept across the shapes a stream can take: a plain answer, a multi-turn tool
loop, a paused run, a Swarm hand-off, and the synchronous ``stream()`` fold. No
mocks: the SDK's own offline models.
"""

from __future__ import annotations

import asyncio

from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer, interrupt
from fastaiagent.agent.swarm import Swarm
from fastaiagent.llm.stream import Paused, StreamDone, TextDelta
from fastaiagent.testing import FunctionModel, TestModel


def _events(stream) -> list:
    async def collect():
        return [e async for e in stream]

    return asyncio.run(collect())


def _weather(city: str) -> str:
    """Current weather for a city."""
    return "18°C and sunny"


def _tool_loop_model():
    calls = {"n": 0}

    def responder(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return "", [{"name": "weather", "arguments": {"city": "Paris"}}]
        return "It is 18°C and sunny."

    return FunctionModel(responder)


def test_a_plain_answer_ends_with_one_stream_done() -> None:
    events = _events(Agent(name="t", llm=TestModel(response="hi")).astream("go"))
    assert isinstance(events[-1], StreamDone)
    assert sum(isinstance(e, StreamDone) for e in events) == 1


def test_a_tool_loop_ends_with_one_stream_done() -> None:
    agent = Agent(
        name="t", llm=_tool_loop_model(), tools=[FunctionTool(name="weather", fn=_weather)]
    )
    events = _events(agent.astream("Weather in Paris?"))
    assert isinstance(events[-1], StreamDone)
    assert sum(isinstance(e, StreamDone) for e in events) == 1
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "It is 18°C and sunny."


def test_a_paused_run_ends_with_paused_not_stream_done(tmp_path) -> None:
    def approve(amount: int) -> str:
        """Needs a human."""
        interrupt(reason="manager_approval", context={"amount": amount})
        return "ok"

    calls = {"n": 0}

    def responder(messages):
        calls["n"] += 1
        return (
            ("", [{"name": "approve", "arguments": {"amount": 5}}]) if calls["n"] == 1 else "done"
        )

    agent = Agent(
        name="t",
        llm=FunctionModel(responder),
        tools=[FunctionTool(name="approve", fn=approve)],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "c.db")),
    )
    events = _events(agent.astream("go", execution_id="p-1"))
    assert isinstance(events[-1], Paused)
    assert not any(isinstance(e, StreamDone) for e in events)


def test_a_swarm_hand_off_ends_with_one_stream_done() -> None:
    first = Agent(
        name="triage",
        llm=FunctionModel(
            lambda ms: ("", [{"name": "handoff_to_billing", "arguments": {"reason": "money"}}])
        ),
    )
    second = Agent(name="billing", llm=TestModel(response="Refund issued."))
    swarm = Swarm(
        name="desk",
        agents=[first, second],
        entrypoint="triage",
        handoffs={"triage": ["billing"], "billing": []},
    )
    events = _events(swarm.astream("I was charged twice"))
    assert isinstance(events[-1], StreamDone)
    assert sum(isinstance(e, StreamDone) for e in events) == 1


def test_stream_still_returns_the_answer() -> None:
    agent = Agent(
        name="t", llm=_tool_loop_model(), tools=[FunctionTool(name="weather", fn=_weather)]
    )
    assert agent.stream("Weather in Paris?").output == "It is 18°C and sunny."
