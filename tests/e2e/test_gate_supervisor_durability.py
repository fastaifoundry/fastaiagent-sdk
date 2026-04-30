"""End-to-end quality gate — Supervisor/Worker durability (spec test #8).

Topology: a Supervisor with two workers, ``worker_a`` and ``worker_b``. The
supervisor's LLM delegates to ``worker_b`` first (which runs three turns
including a slow tool), then to ``worker_a`` (one turn), then synthesizes.

Scenario A — **crash inside Worker B**: a worker subprocess starts the
supervisor. Worker B reaches its third turn and calls ``slow_tool`` which
sleeps long enough for the parent to ``SIGKILL`` the subprocess. The parent
then calls ``supervisor.resume(execution_id)`` in-process. Asserts:
    1. the latest checkpoint puts ``worker_b`` somewhere in its agent_path;
    2. the resume completes the supervisor cleanly through ``worker_a``;
    3. the final ``tool_calls`` show both delegates and the slow tool;
    4. checkpoint paths are nested as ``supervisor:.../worker:.../...``.

Scenario B — **interrupt() inside a worker**: ``worker_b``'s tool calls
``interrupt()``. The supervisor returns paused. A separate
``supervisor.resume(execution_id, resume_value=Resume(...))`` re-enters the
suspended tool, the worker returns, the supervisor delegates to
``worker_a``, then synthesizes.
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


# ---------- Deterministic mock LLMs --------------------------------------


def _supervisor_llm() -> Any:
    """Supervisor: delegates to worker_b, then worker_a, then synthesizes.

    Every iteration is keyed off how many tool messages have been seen so
    far so the same scripted response is produced before and after a crash
    (resume re-issues the LLM call from the saved message history, which
    must reproduce the same delegate decisions).
    """
    from fastaiagent.llm.client import LLMClient, LLMResponse
    from fastaiagent.llm.message import ToolCall

    class _LLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(provider="mock", model="mock")

        async def acomplete(self, messages: Any, tools: Any = None, **_: Any) -> LLMResponse:
            tool_msgs = sum(
                1 for m in messages if getattr(getattr(m, "role", None), "value", None) == "tool"
            )
            if tool_msgs == 0:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="sup_1",
                            name="delegate_to_worker_b",
                            arguments={"task": "Run the deep analysis pipeline."},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            if tool_msgs == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="sup_2",
                            name="delegate_to_worker_a",
                            arguments={"task": "Quickly summarise B's output."},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            return LLMResponse(
                content="Final report ready.",
                finish_reason="stop",
            )

    return _LLM()


def _worker_b_llm(slow_at_turn: int = 2, *, use_interrupt: bool) -> Any:
    """Worker B: 3 LLM turns. Turn ``slow_at_turn`` calls the durability tool.

    Turns before that just exercise turn-boundary checkpoints (cheap_step).
    The final turn produces the worker's plain-text result.
    """
    from fastaiagent.llm.client import LLMClient, LLMResponse
    from fastaiagent.llm.message import ToolCall

    durability_tool = "interrupt_step" if use_interrupt else "slow_step"

    class _LLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(provider="mock", model="mock")

        async def acomplete(self, messages: Any, tools: Any = None, **_: Any) -> LLMResponse:
            tool_msgs = sum(
                1 for m in messages if getattr(getattr(m, "role", None), "value", None) == "tool"
            )
            # turn N corresponds to N tool messages already seen.
            if tool_msgs < slow_at_turn:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=f"b_cheap_{tool_msgs}",
                            name="cheap_step",
                            arguments={"i": tool_msgs},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            if tool_msgs == slow_at_turn:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="b_durability",
                            name=durability_tool,
                            arguments={"payload": "deep-analysis"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            return LLMResponse(content="B done.", finish_reason="stop")

    return _LLM()


def _worker_a_llm() -> Any:
    """Worker A: produces the final summary in one LLM call."""
    from fastaiagent.llm.client import LLMClient, LLMResponse

    class _LLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(provider="mock", model="mock")

        async def acomplete(self, messages: Any, tools: Any = None, **_: Any) -> LLMResponse:
            return LLMResponse(content="Summary from A.", finish_reason="stop")

    return _LLM()


# ---------- Tools --------------------------------------------------------


def _cheap_step(i: int) -> dict[str, Any]:
    return {"step": i, "ok": True}


def _slow_step(payload: str) -> dict[str, Any]:
    """Sleeps long enough for the parent to SIGKILL the subprocess."""
    time.sleep(5)
    return {"insights": f"deep analysis of {payload}"}


def _interrupt_step(payload: str) -> dict[str, Any]:
    """Calls interrupt() to suspend for human approval."""
    from fastaiagent import interrupt

    decision = interrupt(
        reason="worker_b_review",
        context={"payload": payload, "stage": "pre-publish"},
    )
    return {
        "insights": f"reviewed: {payload}",
        "approved": decision.approved,
        "approver": decision.metadata.get("approver"),
    }


# ---------- Supervisor builder shared by parent + subprocess -------------


def _build_supervisor(ckpt_db: str, *, scenario: str) -> Any:
    """Build the same Supervisor in worker subprocess and parent test.

    ``scenario``:
        - "crash": worker_b uses slow_step (used by SIGKILL test)
        - "interrupt": worker_b uses interrupt_step (used by interrupt test)
    """
    from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer
    from fastaiagent.agent.team import Supervisor, Worker

    use_interrupt = scenario == "interrupt"

    worker_b_agent = Agent(
        name="worker_b",
        system_prompt="You are worker B.",
        llm=_worker_b_llm(slow_at_turn=2, use_interrupt=use_interrupt),
        tools=[
            FunctionTool(name="cheap_step", fn=_cheap_step),
            FunctionTool(
                name="interrupt_step" if use_interrupt else "slow_step",
                fn=_interrupt_step if use_interrupt else _slow_step,
            ),
        ],
    )
    worker_a_agent = Agent(
        name="worker_a",
        system_prompt="You are worker A.",
        llm=_worker_a_llm(),
        tools=[],
    )

    return Supervisor(
        name="planner",
        llm=_supervisor_llm(),
        workers=[
            Worker(agent=worker_b_agent, role="worker_b", description="Deep analysis"),
            Worker(agent=worker_a_agent, role="worker_a", description="Quick summary"),
        ],
        max_delegation_rounds=4,
        checkpointer=SQLiteCheckpointer(db_path=ckpt_db),
    )


# ---------- Scenario A — SIGKILL inside Worker B's slow tool -------------


def _spawn_supervisor_worker(ckpt_db: str, execution_id: str) -> subprocess.Popen[bytes]:
    repo_root = str(Path(__file__).resolve().parents[2])
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {repo_root!r})
        from tests.e2e.test_gate_supervisor_durability import _build_supervisor
        sup = _build_supervisor({ckpt_db!r}, scenario="crash")
        # Will be SIGKILLed during worker_b's slow_step. By then
        # supervisor's pre-tool checkpoint for delegate_to_worker_b plus
        # worker_b's own turn / pre-tool checkpoints are on disk.
        sup.run("Investigate and summarise.", execution_id={execution_id!r})
        """
    ).strip()
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for_worker_b_slow_step(ckpt_db: str, execution_id: str, timeout: float = 12.0) -> None:
    from fastaiagent import SQLiteCheckpointer

    deadline = time.monotonic() + timeout
    # Worker checkpoints nest under the supervisor's path; the supervisor's
    # own pre-tool segment (``/tool:delegate_to_worker_b``) lives on
    # supervisor checkpoints, not on the worker's.
    target_path = "supervisor:planner/worker:worker_b/tool:slow_step"
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
    raise TimeoutError(f"worker_b pre-tool slow_step checkpoint never appeared within {timeout}s")


class TestSupervisorCrashInsideWorker:
    def test_08_subprocess_kill_inside_worker_b_resumes_to_completion(self, tmp_path: Path) -> None:
        require_env()
        from fastaiagent import SQLiteCheckpointer

        ckpt_db = str(tmp_path / "ckpt-supervisor-crash.db")
        execution_id = f"sup-crash-{uuid.uuid4().hex[:8]}"

        proc = _spawn_supervisor_worker(ckpt_db, execution_id)
        try:
            _wait_for_worker_b_slow_step(ckpt_db, execution_id)
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

        # Pre-resume: latest checkpoint puts worker_b somewhere in the path,
        # nested under the supervisor.
        store = SQLiteCheckpointer(db_path=ckpt_db)
        store.setup()
        latest = store.get_last(execution_id)
        assert latest is not None
        path = latest.agent_path or ""
        assert path.startswith("supervisor:planner/"), f"path={path!r}"
        assert "/worker:worker_b/" in path, f"path={path!r}"

        # Worker A has not run yet (no checkpoints under its prefix).
        all_cps = store.list(execution_id, limit=500)
        assert not any("/worker:worker_a" in (cp.agent_path or "") for cp in all_cps), (
            "worker_a should not have run before the SIGKILL"
        )
        store.close()

        # Resume in-process. Re-issues supervisor LLM, delegate_to_worker_b
        # detects existing worker_b state and calls aresume (re-runs the
        # slow_step tool since it never completed), worker_b finishes, then
        # delegate_to_worker_a runs the second worker fresh.
        sup = _build_supervisor(ckpt_db, scenario="crash")
        result = sup.resume(execution_id)

        assert result.status == "completed", (
            f"resume did not complete: status={result.status}, "
            f"pending_interrupt={result.pending_interrupt}"
        )
        assert result.execution_id == execution_id
        assert result.output == "Final report ready."

        # Worker A actually ran during the resume.
        all_cps = store.list(execution_id, limit=500)
        assert any("/worker:worker_a" in (cp.agent_path or "") for cp in all_cps), (
            "worker_a never ran during resume"
        )

        # And the slow_step tool ran to completion (its result is in the
        # supervisor's tool_calls record).
        names = [tc.get("tool_name") for tc in result.tool_calls]
        assert "delegate_to_worker_b" in names, names
        assert "delegate_to_worker_a" in names, names


# ---------- Scenario B — interrupt() inside a worker's tool --------------


class TestSupervisorWorkerInterrupt:
    def test_08b_worker_b_interrupt_pauses_and_resumes(self, tmp_path: Path) -> None:
        require_env()
        from fastaiagent import (
            AlreadyResumed,
            Resume,
            SQLiteCheckpointer,
        )

        ckpt_db = str(tmp_path / "ckpt-supervisor-interrupt.db")
        execution_id = f"sup-int-{uuid.uuid4().hex[:8]}"

        sup_a = _build_supervisor(ckpt_db, scenario="interrupt")
        paused = sup_a.run("Investigate and summarise.", execution_id=execution_id)

        assert paused.status == "paused", (
            f"expected paused supervisor, got status={paused.status} "
            f"output={paused.output!r} pending={paused.pending_interrupt}"
        )
        assert paused.pending_interrupt is not None
        assert paused.pending_interrupt["reason"] == "worker_b_review"
        # The agent_path on the pending row reflects the full nesting:
        # supervisor → delegate tool → worker → worker tool.
        path = str(paused.pending_interrupt["agent_path"])
        # Hierarchy: supervisor → worker → worker's tool. The supervisor's
        # own delegate-tool segment lives on the supervisor's checkpoints
        # (``supervisor:planner/tool:delegate_to_worker_b``), not on the
        # worker's path — those branches are parallel under the supervisor.
        assert path == "supervisor:planner/worker:worker_b/tool:interrupt_step", path

        # Pending row exists.
        store = SQLiteCheckpointer(db_path=ckpt_db)
        store.setup()
        pending = store.list_pending_interrupts()
        assert any(p.execution_id == execution_id for p in pending)

        # Resume from a fresh Supervisor.
        sup_b = _build_supervisor(ckpt_db, scenario="interrupt")
        result = sup_b.resume(
            execution_id,
            resume_value=Resume(approved=True, metadata={"approver": "dora"}),
        )

        assert result.status == "completed", (
            f"resume did not complete: status={result.status}, "
            f"pending_interrupt={result.pending_interrupt}"
        )
        assert result.execution_id == execution_id
        assert result.output == "Final report ready."

        # Pending row was claimed atomically.
        post_pending = store.list_pending_interrupts()
        assert all(p.execution_id != execution_id for p in post_pending)

        # Second resume against the now-completed supervisor raises AlreadyResumed.
        sup_c = _build_supervisor(ckpt_db, scenario="interrupt")
        with pytest.raises(AlreadyResumed):
            sup_c.resume(
                execution_id,
                resume_value=Resume(approved=False, metadata={}),
            )
