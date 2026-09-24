"""A paused run ends ``astream()`` cleanly with a ``Paused`` event (1.77.0, audit M4).

Before 1.77.0 a tool that paused the run — a managed approval policy, or an
``interrupt()`` — made the private ``_AgentInterrupted`` escape the generator: the
caller got an exception it could not import, instead of the pause ``arun()``
returns. The pause was already checkpointed; only the stream failed to say so.

Real ``Agent`` / ``Swarm``, real ``SQLiteCheckpointer``; the model is the repo's
scripted ``MockLLMClient`` (what it says is irrelevant — only the event sequence is).
The policy case runs against the local stand-in plane (``tests/_governance_plane.py``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, Paused, interrupt
from fastaiagent.chain.interrupt import Resume
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.llm.stream import StreamEvent, TextDelta
from tests._governance_plane import AGENT_ID, GovPlane, reset_connection, serve
from tests.conftest import MockLLMClient


def _llm(tool: str, args: dict[str, Any]) -> MockLLMClient:
    return MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name=tool, arguments=args)],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done", finish_reason="stop"),
        ]
    )


def _sign_off_tool() -> FunctionTool:
    def ask_human(question: str) -> str:
        decision = interrupt(reason="need_sign_off", context={"question": question})
        return "signed" if decision.approved else "declined"

    return FunctionTool(name="ask_human", fn=ask_human)


async def _collect(stream: Any) -> list[StreamEvent]:
    return [event async for event in stream]


def test_an_interrupt_ends_the_stream_with_a_paused_event(tmp_path: Path) -> None:
    agent = Agent(
        name="clerk",
        llm=_llm("ask_human", {"question": "ok?"}),
        tools=[_sign_off_tool()],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )

    events = asyncio.run(_collect(agent.astream("Get sign-off.", execution_id="run-s")))

    paused = events[-1]
    assert isinstance(paused, Paused), events
    assert paused.reason == "need_sign_off"
    assert paused.execution_id == "run-s"
    assert paused.context == {"question": "ok?"}
    assert [e for e in events if isinstance(e, Paused)] == [paused]  # exactly one, last
    # It is a real pause: the run resumes like one returned by arun().
    final = asyncio.run(agent.aresume("run-s", resume_value=Resume(approved=True)))
    assert final.status == "completed", final


@pytest.fixture
def plane(isolated_local_db: Path) -> Iterator[GovPlane]:
    import fastaiagent

    with serve() as (state, url):
        fastaiagent.connect(api_key="fa_k_stream_test", target=url)
        try:
            yield state
        finally:
            reset_connection()


def test_a_policy_pause_streams_the_tool_and_its_arguments(plane: GovPlane, tmp_path: Path) -> None:
    ran: list[dict[str, Any]] = []

    def transfer_funds(amount: int, to: str) -> str:
        ran.append({"amount": amount, "to": to})
        return "Transferred."

    agent = Agent(
        name="banker",
        agent_id=AGENT_ID,
        llm=_llm("transfer_funds", {"amount": 500, "to": "Bob"}),
        tools=[FunctionTool(name="transfer_funds", fn=transfer_funds)],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )

    events = asyncio.run(_collect(agent.astream("Transfer $500 to Bob.", execution_id="run-s")))

    paused = events[-1]
    assert isinstance(paused, Paused), events
    assert paused.reason == "policy_approval_required"
    assert paused.context["tool"] == "transfer_funds"
    assert paused.context["tool_input"] == {"amount": 500, "to": "Bob"}
    assert paused.context["pending_id"] == "pr-run-s"
    assert ran == []


def test_a_swarm_stream_forwards_the_pause_and_stops(tmp_path: Path) -> None:
    from fastaiagent.agent.swarm import Swarm

    swarm = Swarm(
        name="desk",
        agents=[
            Agent(
                name="clerk", llm=_llm("ask_human", {"question": "ok?"}), tools=[_sign_off_tool()]
            )
        ],
        entrypoint="clerk",
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )

    events = asyncio.run(_collect(swarm.astream("Get sign-off.")))

    assert isinstance(events[-1], Paused), events
    assert events[-1].reason == "need_sign_off"
    assert not any(isinstance(e, TextDelta) and e.text == "done" for e in events)
