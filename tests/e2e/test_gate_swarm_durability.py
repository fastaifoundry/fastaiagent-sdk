"""End-to-end quality gate — Swarm durability (spec test #7).

A 3-agent swarm: ``researcher`` → ``analyst`` → ``reporter``.

Scenario A — **crash mid-handoff**: a worker subprocess starts the swarm.
``researcher`` hands off to ``analyst``. ``analyst`` runs a slow tool that
sleeps long enough for the parent to ``SIGKILL`` the worker. The parent
then calls ``swarm.resume(execution_id)`` in-process and verifies:
    1. the latest checkpoint identifies ``analyst`` as the active agent;
    2. the resume completes the swarm cleanly through ``reporter``;
    3. the final ``handoff_history`` reflects researcher → analyst → reporter.

Scenario B — **interrupt() inside a swarm agent**: ``analyst``'s tool calls
``interrupt()``. The swarm returns paused. A separate
``swarm.resume(execution_id, resume_value=Resume(...))`` re-enters the
suspended tool, the analyst returns, and the swarm hands off to
``reporter`` for the final answer.
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


# ---------- Deterministic mock LLMs for the three agents -----------------


def _researcher_llm() -> Any:
    """Researcher: emits a handoff_to_analyst tool call on turn 0."""
    from fastaiagent.llm.client import LLMClient, LLMResponse
    from fastaiagent.llm.message import ToolCall

    class _LLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(provider="mock", model="mock")
            self._n = 0

        async def acomplete(self, messages: Any, tools: Any = None, **_: Any) -> LLMResponse:
            self._n += 1
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"r_{self._n}",
                        name="handoff_to_analyst",
                        arguments={"reason": "research done"},
                    )
                ],
                finish_reason="tool_calls",
            )

    return _LLM()


def _analyst_llm(use_slow: bool, use_interrupt: bool) -> Any:
    """Analyst: turn 0 calls a domain tool, turn 1 hands off to reporter."""
    from fastaiagent.llm.client import LLMClient, LLMResponse
    from fastaiagent.llm.message import ToolCall

    domain_tool = "slow_analyze" if use_slow else "interrupt_analyze" if use_interrupt else "noop"

    class _LLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(provider="mock", model="mock")
            self._n = 0

        async def acomplete(self, messages: Any, tools: Any = None, **_: Any) -> LLMResponse:
            self._n += 1
            tool_msgs = sum(
                1 for m in messages if getattr(getattr(m, "role", None), "value", None) == "tool"
            )
            if tool_msgs == 0:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=f"a_{self._n}",
                            name=domain_tool,
                            arguments={"facts": "found things"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"a_{self._n}",
                        name="handoff_to_reporter",
                        arguments={"reason": "analysis done"},
                    )
                ],
                finish_reason="tool_calls",
            )

    return _LLM()


def _reporter_llm() -> Any:
    """Reporter: produces the final swarm answer."""
    from fastaiagent.llm.client import LLMClient, LLMResponse

    class _LLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(provider="mock", model="mock")

        async def acomplete(self, messages: Any, tools: Any = None, **_: Any) -> LLMResponse:
            return LLMResponse(
                content="Report complete.",
                finish_reason="stop",
            )

    return _LLM()


# ---------- Tools -------------------------------------------------------


def _slow_analyze(facts: str) -> dict[str, Any]:
    """Sleeps long enough for the parent to SIGKILL mid-execution."""
    time.sleep(5)
    return {"insights": f"deep analysis of {facts}", "ok": True}


def _interrupt_analyze(facts: str) -> dict[str, Any]:
    """Calls interrupt() to suspend for human approval."""
    from fastaiagent import interrupt

    decision = interrupt(
        reason="analyst_review",
        context={"facts": facts, "stage": "pre-publish"},
    )
    return {
        "insights": f"reviewed: {facts}",
        "approved": decision.approved,
        "approver": decision.metadata.get("approver"),
    }


def _noop_analyze(facts: str) -> dict[str, Any]:
    """Plain happy-path tool used by the legacy success test."""
    return {"insights": f"summary of {facts}"}


# ---------- Swarm builder shared by parent + subprocess ------------------


def _build_swarm(ckpt_db: str, *, scenario: str) -> Any:
    """Build the same swarm in worker and parent.

    ``scenario`` selects which analyst tool is wired up:
        - "crash": slow_analyze (used by SIGKILL test)
        - "interrupt": interrupt_analyze (used by interrupt test)
    """
    from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer, Swarm

    if scenario == "crash":
        analyst_tool = FunctionTool(name="slow_analyze", fn=_slow_analyze)
        analyst_llm_obj = _analyst_llm(use_slow=True, use_interrupt=False)
    elif scenario == "interrupt":
        analyst_tool = FunctionTool(name="interrupt_analyze", fn=_interrupt_analyze)
        analyst_llm_obj = _analyst_llm(use_slow=False, use_interrupt=True)
    else:
        analyst_tool = FunctionTool(name="noop", fn=_noop_analyze)
        analyst_llm_obj = _analyst_llm(use_slow=False, use_interrupt=False)

    researcher = Agent(
        name="researcher",
        system_prompt="You are a researcher.",
        llm=_researcher_llm(),
        tools=[],
    )
    analyst = Agent(
        name="analyst",
        system_prompt="You are an analyst.",
        llm=analyst_llm_obj,
        tools=[analyst_tool],
    )
    reporter = Agent(
        name="reporter",
        system_prompt="You are a reporter.",
        llm=_reporter_llm(),
        tools=[],
    )

    return Swarm(
        name="content_team",
        agents=[researcher, analyst, reporter],
        entrypoint="researcher",
        handoffs={
            "researcher": ["analyst"],
            "analyst": ["reporter"],
            "reporter": [],
        },
        max_handoffs=4,
        checkpointer=SQLiteCheckpointer(db_path=ckpt_db),
    )


# ---------- Scenario A — SIGKILL mid-handoff -----------------------------


def _spawn_swarm_worker(ckpt_db: str, execution_id: str) -> subprocess.Popen[bytes]:
    repo_root = str(Path(__file__).resolve().parents[2])
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {repo_root!r})
        from tests.e2e.test_gate_swarm_durability import _build_swarm
        swarm = _build_swarm({ckpt_db!r}, scenario="crash")
        # Will be SIGKILLed during analyst's slow_analyze tool. By then
        # researcher's handoff and the swarm's handoff:1 (active=analyst)
        # checkpoint are already on disk.
        swarm.run("Investigate the topic.", execution_id={execution_id!r})
        """
    ).strip()
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for_analyst_pre_tool(ckpt_db: str, execution_id: str, timeout: float = 10.0) -> None:
    """Poll until analyst's pre-tool checkpoint is committed."""
    from fastaiagent import SQLiteCheckpointer

    deadline = time.monotonic() + timeout
    target_path = "swarm:content_team/agent:analyst/tool:slow_analyze"
    while time.monotonic() < deadline:
        try:
            cp = SQLiteCheckpointer(db_path=ckpt_db)
            cp.setup()
            latest = cp.get_last(execution_id)
            cp.close()
        except Exception:
            latest = None
        if latest is not None and (latest.agent_path or "") == target_path:
            return
        time.sleep(0.05)
    raise TimeoutError(f"analyst pre-tool checkpoint never appeared within {timeout}s")


class TestSwarmCrashMidHandoff:
    def test_07_subprocess_kill_mid_analyst_resumes_to_completion(self, tmp_path: Path) -> None:
        require_env()
        from fastaiagent import SQLiteCheckpointer

        ckpt_db = str(tmp_path / "ckpt-swarm-crash.db")
        execution_id = f"swarm-crash-{uuid.uuid4().hex[:8]}"

        proc = _spawn_swarm_worker(ckpt_db, execution_id)
        try:
            _wait_for_analyst_pre_tool(ckpt_db, execution_id)
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
            assert proc.returncode is not None and proc.returncode < 0, (
                f"worker did not die on SIGKILL: returncode={proc.returncode} "
                f"stderr={proc.stderr.read() if proc.stderr else b''!r}"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

        # Pre-resume sanity: latest checkpoint puts analyst as the active
        # agent (agent_path starts with the analyst prefix).
        store = SQLiteCheckpointer(db_path=ckpt_db)
        store.setup()
        latest = store.get_last(execution_id)
        assert latest is not None
        assert (latest.agent_path or "").startswith("swarm:content_team/agent:analyst")
        # The handoff:1 boundary records analyst as active.
        cps = store.list(execution_id, limit=500)
        handoff1 = next(
            (cp for cp in cps if cp.node_id == "handoff:1"),
            None,
        )
        assert handoff1 is not None
        assert handoff1.state_snapshot.get("active_agent") == "analyst"
        store.close()

        # Resume in-process. Re-runs analyst from scratch (crash recovery
        # re-issues the LLM call), then hands off to reporter.
        swarm = _build_swarm(ckpt_db, scenario="crash")
        result = swarm.resume(execution_id)

        assert result.status == "completed", (
            f"resume did not complete: status={result.status}, "
            f"pending_interrupt={result.pending_interrupt}"
        )
        assert result.execution_id == execution_id
        assert result.output == "Report complete."

        # The accumulated tool_calls record both handoffs (researcher→analyst,
        # analyst→reporter) plus analyst's domain tool.
        names = [tc.get("tool_name") for tc in result.tool_calls]
        assert "handoff_to_analyst" in names, f"tool_calls: {names}"
        assert "handoff_to_reporter" in names, f"tool_calls: {names}"
        assert "slow_analyze" in names, f"tool_calls: {names}"


# ---------- Scenario B — interrupt() inside a swarm agent ---------------


class TestSwarmInterruptInsideAgent:
    def test_07b_analyst_tool_interrupt_pauses_and_resumes(self, tmp_path: Path) -> None:
        require_env()
        from fastaiagent import (
            AlreadyResumed,
            Resume,
            SQLiteCheckpointer,
        )

        ckpt_db = str(tmp_path / "ckpt-swarm-interrupt.db")
        execution_id = f"swarm-int-{uuid.uuid4().hex[:8]}"

        swarm_a = _build_swarm(ckpt_db, scenario="interrupt")
        paused = swarm_a.run("Analyze the data.", execution_id=execution_id)

        assert paused.status == "paused", (
            f"expected paused swarm, got status={paused.status} "
            f"output={paused.output!r} pending={paused.pending_interrupt}"
        )
        assert paused.pending_interrupt is not None
        assert paused.pending_interrupt["reason"] == "analyst_review"
        # The agent_path on the pending row reflects the full nesting:
        # swarm → agent → tool.
        assert paused.pending_interrupt["agent_path"] == (
            "swarm:content_team/agent:analyst/tool:interrupt_analyze"
        )

        # The pending row is observable in the checkpointer.
        store = SQLiteCheckpointer(db_path=ckpt_db)
        store.setup()
        pending = store.list_pending_interrupts()
        assert any(p.execution_id == execution_id for p in pending)

        # Resume from a fresh Swarm (production shape: pause + resume in
        # different processes).
        swarm_b = _build_swarm(ckpt_db, scenario="interrupt")
        result = swarm_b.resume(
            execution_id,
            resume_value=Resume(approved=True, metadata={"approver": "carol"}),
        )

        assert result.status == "completed", (
            f"resume did not complete: status={result.status}, "
            f"pending_interrupt={result.pending_interrupt}"
        )
        assert result.execution_id == execution_id
        assert result.output == "Report complete."

        # Pending row was atomically claimed.
        post_pending = store.list_pending_interrupts()
        assert all(p.execution_id != execution_id for p in post_pending)

        # Second resume against the now-completed swarm raises AlreadyResumed.
        swarm_c = _build_swarm(ckpt_db, scenario="interrupt")
        with pytest.raises(AlreadyResumed):
            swarm_c.resume(
                execution_id,
                resume_value=Resume(approved=False, metadata={}),
            )
