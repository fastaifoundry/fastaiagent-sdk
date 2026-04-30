"""End-to-end quality gate — single-Agent durability (spec test #6).

Two scenarios, both backed by a real ``SQLiteCheckpointer``.

A. **Crash mid-tool** (subprocess + SIGKILL): a worker subprocess runs an
   Agent whose tool sleeps long enough for the parent to ``SIGKILL`` it.
   The parent then calls ``agent.resume(execution_id)`` in-process and
   verifies the agent finishes — proving the pre-tool checkpoint is enough
   to recover from a true process crash without re-issuing the LLM call.

B. **Tool calls interrupt()**: a tool calls :func:`interrupt` to suspend
   for human approval. ``agent.run`` returns ``AgentResult(status="paused")``
   with the pending-interrupt payload. A separate
   ``agent.resume(execution_id, resume_value=Resume(approved=True))``
   re-enters the suspended tool — ``interrupt()`` returns the resume value —
   the LLM gets the tool result and produces the final answer.

The mock LLM is deterministic across both processes (no real provider calls).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


# ---------- Shared agent + tool definitions used by both scenarios -------


def _scripted_llm(tool_name: str, tool_args: dict[str, Any], final_text: str) -> Any:
    """Deterministic LLM: turn 0 → call ``tool_name`` with ``tool_args``;
    turn 1 → return ``final_text``. The same script runs in worker + parent.
    """
    from fastaiagent.llm.client import LLMClient, LLMResponse
    from fastaiagent.llm.message import ToolCall

    class _MockLLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(provider="mock", model="mock")
            self._call_count = 0

        async def acomplete(self, messages: Any, tools: Any = None, **_: Any) -> LLMResponse:
            self._call_count += 1
            tool_message_count = sum(
                1
                for m in messages
                if getattr(m, "role", None) is not None and getattr(m.role, "value", None) == "tool"
            )
            if tool_message_count == 0:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=f"call_{self._call_count}",
                            name=tool_name,
                            arguments=tool_args,
                        )
                    ],
                    finish_reason="tool_calls",
                )
            return LLMResponse(content=final_text, finish_reason="stop")

    return _MockLLM()


def _slow_paid(amount: int) -> dict[str, Any]:
    """Tool used by the crash test — sleeps long enough to be SIGKILLed."""
    time.sleep(5)
    return {"ok": True, "amount": amount}


def _build_crash_agent(ckpt_db: str) -> Any:
    """Build the same Agent (with the slow tool) in the worker and parent."""
    from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer

    return Agent(
        name="crash-recovery-agent",
        system_prompt="You are a tester.",
        llm=_scripted_llm("slow_paid", {"amount": 100}, "Done — paid $100."),
        tools=[FunctionTool(name="slow_paid", fn=_slow_paid)],
        checkpointer=SQLiteCheckpointer(db_path=ckpt_db),
    )


# ---------- Scenario A — SIGKILL mid-tool, resume completes the agent ----


def _spawn_agent_worker(ckpt_db: str, execution_id: str) -> subprocess.Popen[bytes]:
    repo_root = str(Path(__file__).resolve().parents[2])
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {repo_root!r})
        from tests.e2e.test_gate_agent_durability import _build_crash_agent
        agent = _build_crash_agent({ckpt_db!r})
        # Will be killed mid-slow_paid by the parent. The pre-tool
        # checkpoint will already be on disk by then.
        agent.run("Pay $100", execution_id={execution_id!r}, trace=False)
        """
    ).strip()
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for_pre_tool_checkpoint(
    ckpt_db: str, execution_id: str, tool_name: str, timeout: float = 10.0
) -> None:
    from fastaiagent import SQLiteCheckpointer

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            cp = SQLiteCheckpointer(db_path=ckpt_db)
            cp.setup()
            latest = cp.get_last(execution_id)
            cp.close()
        except Exception:
            latest = None
        if latest is not None and latest.node_id == f"turn:0/tool:{tool_name}":
            return
        time.sleep(0.05)
    raise TimeoutError(f"pre-tool checkpoint never appeared for {execution_id!r} within {timeout}s")


class TestAgentCrashMidTool:
    def test_06a_subprocess_kill_resumes_at_pre_tool_boundary(self, tmp_path: Path) -> None:
        require_env()
        from fastaiagent import SQLiteCheckpointer

        ckpt_db = str(tmp_path / "ckpt-agent-crash.db")
        execution_id = f"agent-crash-{uuid.uuid4().hex[:8]}"

        proc = _spawn_agent_worker(ckpt_db, execution_id)
        try:
            _wait_for_pre_tool_checkpoint(ckpt_db, execution_id, "slow_paid")
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
            assert proc.returncode is not None
            assert proc.returncode < 0, (
                f"worker did not die on SIGKILL: returncode={proc.returncode}, "
                f"stderr={proc.stderr.read() if proc.stderr else b''!r}"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

        # Pre-resume: latest checkpoint is the pre-tool boundary.
        store = SQLiteCheckpointer(db_path=ckpt_db)
        store.setup()
        latest = store.get_last(execution_id)
        assert latest is not None
        assert latest.node_id == "turn:0/tool:slow_paid"
        assert latest.status == "completed"
        assert latest.state_snapshot.get("tool_name") == "slow_paid"
        assert latest.node_input == {"amount": 100}
        store.close()

        # In-process resume — re-invokes slow_paid (same args), then the
        # mock LLM produces the final answer.
        agent = _build_crash_agent(ckpt_db)
        result = agent.resume(execution_id, trace=False)

        assert result.status == "completed", (
            f"resume did not complete: status={result.status}, "
            f"pending_interrupt={result.pending_interrupt}"
        )
        assert result.execution_id == execution_id
        assert result.output == "Done — paid $100."


# ---------- Scenario B — tool calls interrupt(), resume injects Resume ----


class TestAgentToolInterrupt:
    def test_06b_tool_interrupt_pauses_and_resumes(self, tmp_path: Path) -> None:
        require_env()
        from fastaiagent import (
            Agent,
            AlreadyResumed,
            FunctionTool,
            Resume,
            SQLiteCheckpointer,
            interrupt,
        )

        ckpt_db = str(tmp_path / "ckpt-agent-interrupt.db")

        attempts = {"count": 0}

        def approve_refund(amount: int) -> dict[str, Any]:
            attempts["count"] += 1
            decision = interrupt(
                reason="manager_approval",
                context={"amount": amount, "reason": "high-value refund"},
            )
            return {
                "approved": decision.approved,
                "approver": decision.metadata.get("approver"),
                "amount": amount,
            }

        def _make_agent() -> Agent:
            return Agent(
                name="refund-agent",
                system_prompt="You are a refund agent.",
                llm=_scripted_llm(
                    "approve_refund",
                    {"amount": 50_000},
                    "Refund approved by alice for $50000.",
                ),
                tools=[FunctionTool(name="approve_refund", fn=approve_refund)],
                checkpointer=SQLiteCheckpointer(db_path=ckpt_db),
            )

        execution_id = f"agent-int-{uuid.uuid4().hex[:8]}"

        agent_a = _make_agent()
        paused = agent_a.run("Refund $50000", execution_id=execution_id, trace=False)

        assert paused.status == "paused", (
            f"expected paused, got status={paused.status} output={paused.output!r} "
            f"pending={paused.pending_interrupt}"
        )
        assert paused.pending_interrupt is not None
        assert paused.pending_interrupt["reason"] == "manager_approval"
        assert paused.pending_interrupt["context"] == {
            "amount": 50_000,
            "reason": "high-value refund",
        }
        assert paused.pending_interrupt["agent_path"] == ("agent:refund-agent/tool:approve_refund")
        assert attempts["count"] == 1

        # Pending row is visible in the checkpointer.
        store = SQLiteCheckpointer(db_path=ckpt_db)
        store.setup()
        pending = store.list_pending_interrupts()
        assert any(p.execution_id == execution_id for p in pending)

        # Resume from a fresh Agent — production shape: pause and resume in
        # different processes / different objects.
        agent_b = _make_agent()
        result = agent_b.resume(
            execution_id,
            resume_value=Resume(approved=True, metadata={"approver": "alice"}),
            trace=False,
        )

        assert result.status == "completed", (
            f"resume did not complete: status={result.status}, "
            f"pending_interrupt={result.pending_interrupt}"
        )
        assert result.execution_id == execution_id
        assert result.output == "Refund approved by alice for $50000."

        # Pending row was atomically claimed.
        post_pending = store.list_pending_interrupts()
        assert all(p.execution_id != execution_id for p in post_pending)

        # Second resume against the now-completed run raises AlreadyResumed.
        agent_c = _make_agent()
        with pytest.raises(AlreadyResumed):
            agent_c.resume(
                execution_id,
                resume_value=Resume(approved=False, metadata={}),
                trace=False,
            )
