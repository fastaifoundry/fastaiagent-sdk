"""Resume and fork record the resumed question in memory (1.81.0).

``aresume`` and ``_fork_branch`` passed ``_arun_core`` the *first* user message
in the checkpoint as "the input", and ``_arun_core`` writes that to memory. But
the checkpointed prompt starts with the memory window, so the first user message
is an old turn: resuming "TURN-2" recorded "TURN-1" again. The re-ask for a
structured-output failure also appended its correction prompt to the list the
loop-end checkpoint saves, so a later resume could record *that* as the user's
question.

No mocking of the SDK: a real ``SQLiteCheckpointer``, real ``interrupt()`` /
``Resume``, the repo's scripted ``MockLLMClient``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from fastaiagent import Agent, AgentConfig, AgentMemory, FunctionTool, interrupt
from fastaiagent.chain.interrupt import Resume
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
from fastaiagent.guardrail import Guardrail, GuardrailPosition
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import AssistantMessage, MessageRole, ToolCall, UserMessage
from tests.conftest import MockLLMClient


def _seeded_memory() -> AgentMemory:
    memory = AgentMemory()
    memory.add(UserMessage("TURN-1: hello"))
    memory.add(AssistantMessage("hi there"))
    return memory


def _user_turns(memory: AgentMemory) -> list[str]:
    return [m.content for m in memory.messages if m.role == MessageRole.user]


def _store(tmp_path: Path) -> SQLiteCheckpointer:
    store = SQLiteCheckpointer(str(tmp_path / "ckpt.db"))
    store.setup()
    return store


def test_resume_records_the_resumed_question(tmp_path):
    def ask_human(question: str) -> str:
        decision = interrupt(reason="need_sign_off", context={"question": question})
        return "signed" if decision.approved else "declined"

    llm = MockLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="ask_human", arguments={"question": "ok?"})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="refund approved", finish_reason="stop"),
        ]
    )
    memory = _seeded_memory()
    agent = Agent(
        name="clerk",
        llm=llm,
        tools=[FunctionTool(name="ask_human", fn=ask_human)],
        memory=memory,
        checkpointer=_store(tmp_path),
    )

    paused = asyncio.run(agent.arun("TURN-2: please refund order Z9", execution_id="run-1"))
    assert paused.status == "paused"
    final = asyncio.run(agent.aresume("run-1", resume_value=Resume(approved=True)))

    assert final.output == "refund approved"
    assert _user_turns(memory) == ["TURN-1: hello", "TURN-2: please refund order Z9"]


def test_fork_records_the_forked_question(tmp_path):
    memory = _seeded_memory()
    agent = Agent(
        name="forker",
        llm=MockLLMClient([LLMResponse(content="answer", finish_reason="stop")]),
        memory=memory,
        checkpointer=_store(tmp_path),
    )
    asyncio.run(agent.arun("TURN-2: what is my balance?", execution_id="src"))
    asyncio.run(agent.afork("src"))

    assert _user_turns(memory)[-1] == "TURN-2: what is my balance?"


class _Answer(BaseModel):
    text: str


@pytest.mark.asyncio
async def test_a_resume_after_a_structured_reask_records_the_question(tmp_path):
    """The re-ask's correction prompt must not become 'the user's question'."""
    llm = MockLLMClient(
        [
            LLMResponse(content="not json", finish_reason="stop"),
            LLMResponse(content='{"text": "bad"}', finish_reason="stop"),  # re-ask: parses
            LLMResponse(content='{"text": "good"}', finish_reason="stop"),  # after resume
        ]
    )
    memory = AgentMemory()
    agent = Agent(
        name="structured",
        llm=llm,
        output_type=_Answer,
        config=AgentConfig(output_retries=1),
        guardrails=[
            Guardrail(
                name="no_bad",
                position=GuardrailPosition.output,
                blocking=True,
                fn=lambda text: "bad" not in text,
            )
        ],
        memory=memory,
        checkpointer=_store(tmp_path),
    )

    with pytest.raises(Exception):
        await agent.arun("What is the answer?", execution_id="ex-reask", trace=False)
    await agent.aresume("ex-reask")

    assert _user_turns(memory) == ["What is the answer?"]
