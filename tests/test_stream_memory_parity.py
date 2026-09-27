"""``astream`` stores the same turn in memory that ``run`` does (1.82.0).

* ``astream`` added up the text of every turn of the tool loop, so a model that
  said "Let me check." before calling a tool had that preamble stored as part of
  its answer. ``run`` stores only the final reply.
* With ``RedactPII``, ``run`` stored the redacted reply (the one it returns) and
  ``astream`` the raw streamed text. Both now store the reply after all
  middleware; the user's message is stored as they said it.

No mocking: the SDK's offline ``FunctionModel``, a real tool, real middleware.
"""

from __future__ import annotations

import asyncio

import pytest

from fastaiagent import Agent, AgentMemory, FunctionTool, RedactPII
from fastaiagent.llm.message import MessageRole
from fastaiagent.testing import FunctionModel


def _lookup() -> str:
    """Look the answer up."""
    return "42"


def _tool_then_answer() -> FunctionModel:
    def responder(messages):
        if any(m.role == MessageRole.tool for m in messages):
            return "The answer is 42."
        return ("Let me check. ", [{"name": "_lookup", "arguments": {}}])

    return FunctionModel(responder)


def _stored(memory: AgentMemory) -> list[tuple[str, str]]:
    return [(m.role.value, m.content or "") for m in memory.messages]


def _run(agent: Agent, text: str, streamed: bool) -> None:
    if not streamed:
        agent.run(text)
        return

    async def drain() -> None:
        async for _ in agent.astream(text):
            pass

    asyncio.run(drain())


@pytest.mark.parametrize("streamed", [False, True], ids=["run", "astream"])
def test_the_stored_reply_is_the_final_one(streamed):
    memory = AgentMemory()
    agent = Agent(
        name="a",
        llm=_tool_then_answer(),
        tools=[FunctionTool(name="_lookup", fn=_lookup)],
        memory=memory,
    )
    _run(agent, "what is the answer?", streamed)
    assert _stored(memory) == [("user", "what is the answer?"), ("assistant", "The answer is 42.")]


@pytest.mark.parametrize("streamed", [False, True], ids=["run", "astream"])
def test_redact_pii_stores_the_redacted_reply_and_the_users_words(streamed):
    memory = AgentMemory()
    agent = Agent(
        name="a",
        llm=FunctionModel(lambda messages: "Sure, write to bob@example.com."),
        middleware=[RedactPII()],
        memory=memory,
    )
    _run(agent, "My email is carol@example.com", streamed)
    stored = dict(_stored(memory))
    assert stored["user"] == "My email is carol@example.com"  # as the user said it
    assert "bob@example.com" not in stored["assistant"]  # the reply after middleware
    assert memory.messages[-1].role == MessageRole.assistant
