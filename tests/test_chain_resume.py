"""Resume-contract tests for ``Chain.resume`` / ``Chain.aresume``.

Covers ``ChainResumeError`` raised when the caller's ``resume_value``
intent and the checkpoint's status disagree — the spec for resume is
documented in ``docs/chains/spec.md`` §Resume.

``ChainResumeError`` subclasses ``ChainCheckpointError`` so existing
``except ChainCheckpointError:`` handlers in v1.13.x stay compatible.
"""

from __future__ import annotations

import pytest

from fastaiagent.agent import Agent
from fastaiagent.chain import Chain, NodeType
from fastaiagent.chain.interrupt import AlreadyResumed, Resume, interrupt
from fastaiagent.checkpointers import SQLiteCheckpointer
from fastaiagent.llm.client import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    """Deterministic LLM stand-in (real subclass — not unittest.mock)."""

    def __init__(self, response_text: str = "ok") -> None:
        super().__init__(provider="mock", model="mock")
        self._response_text = response_text

    async def acomplete(self, messages, tools=None, **kwargs):
        return LLMResponse(content=self._response_text, finish_reason="stop")


def _agent(name: str, response: str = "ok") -> Agent:
    return Agent(name=name, llm=MockLLMClient(response), system_prompt="test")


@pytest.fixture
def checkpointer(tmp_path):
    """SQLite checkpointer scoped to the test's tmpdir — keeps tests isolated."""
    cp = SQLiteCheckpointer(db_path=str(tmp_path / "ckpt.db"))
    cp.setup()
    return cp


class TestResumeValueContract:
    """``resume_value`` is only valid on interrupted checkpoints.
    Mismatched intent raises ``ChainResumeError``.
    """

    @pytest.mark.asyncio
    async def test_resume_value_on_completed_checkpoint_raises_already_resumed(
        self, checkpointer
    ):
        # When a chain has no pending interrupt (either because it never
        # interrupted, or because a prior resume already claimed the row),
        # passing resume_value surfaces :class:`AlreadyResumed`. This is the
        # long-standing v1.x contract and the path the UI/CLI rely on to
        # return 409 / "already resumed" responses.
        chain = Chain("done", checkpointer=checkpointer)
        chain.add_node("a", agent=_agent("a", "alpha"))
        chain.add_node("b", agent=_agent("b", "beta"))
        chain.connect("a", "b")

        result = await chain.aexecute({"input": "hi"})
        assert result.status == "completed"

        with pytest.raises(AlreadyResumed):
            await chain.aresume(result.execution_id, resume_value=Resume(approved=True))

    @pytest.mark.asyncio
    async def test_interrupted_resume_without_value_raises(self, checkpointer):
        from fastaiagent._internal.errors import ChainResumeError

        # hitl_handler signature is ``(node, context, state)`` — see
        # executor._execute_node. We use it to trigger ``interrupt()``
        # which the executor catches and persists as "interrupted".
        def _gate(_node, _ctx, _state):
            interrupt(reason="need approval", context={})
            return True  # never reached on the first call

        chain = Chain("gated", checkpointer=checkpointer)
        chain.add_node("a", agent=_agent("a"))
        chain.add_node("gate", type=NodeType.hitl)
        chain.add_node("b", agent=_agent("b"))
        chain.connect("a", "gate")
        chain.connect("gate", "b")

        result = await chain.aexecute({"input": "hi"}, hitl_handler=_gate)
        assert result.status == "paused"

        # Forgetting to pass resume_value on an interrupted checkpoint is
        # the classic mistake — must surface with a clear error.
        with pytest.raises(ChainResumeError, match="pass resume_value"):
            await chain.aresume(result.execution_id)

    @pytest.mark.asyncio
    async def test_chainresumeerror_is_chaincheckpointerror_subclass(self):
        """Backwards-compat: existing ``except ChainCheckpointError:`` keeps catching."""
        from fastaiagent._internal.errors import (
            ChainCheckpointError,
            ChainResumeError,
        )

        assert issubclass(ChainResumeError, ChainCheckpointError)
