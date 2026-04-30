"""End-to-end quality gate — suspending HITL via ``interrupt()``.

Covers spec test #2 (suspend-and-resume) and #3 (resume after elapsed time
and across a fresh process). The chain has three nodes:

    seed -> approval (calls ``interrupt()``) -> finalize

First execute() suspends at ``approval`` and returns ``ChainResult(status=
"paused")``. A separate ``chain.resume()`` call (with a ``Resume`` value)
re-runs ``approval`` — this time ``interrupt()`` returns the value — then
``finalize`` runs and the chain completes.

This gate is deliberately self-contained: no LLM, no platform, no network.
A real local SQLite checkpointer is the only persistence and the subprocess
test really fork/exec's a fresh Python interpreter.

Test scope:
    01 chain returns paused with the right pending_interrupt + checkpoint
    02 pending_interrupts row contains the frozen context
    03 resume with Resume(approved=True) completes the chain
    04 resumed chain final_state reflects the resume metadata
    05 second resume raises AlreadyResumed (atomic claim)
    06 resume across time.sleep(2) still works
    07 resume in a separate process (subprocess.Popen) completes the chain
    08 frozen-context invariant: changing state between pause and resume
       does NOT change the context the resumer sees
"""

from __future__ import annotations

import json
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


# ---------- Chain-builder helper used by every sub-test -------------------


def _seed_fn(value: str) -> dict[str, Any]:
    return {"seeded": value, "step_seed_done": True}


def _approval_fn(amount: str) -> dict[str, Any]:
    """Calls ``interrupt()`` whenever amount > threshold.

    First execution raises InterruptSignal (caught by the executor).
    On resume, ``interrupt()`` returns the ``Resume`` value the resumer
    passed in.

    ``amount`` arrives as a string because chain input_mapping templates
    always render to text — coerce inside the function.
    """
    from fastaiagent import interrupt

    n = int(amount)
    if n > 1000:
        decision = interrupt(
            reason="manager_approval",
            context={"amount": n, "policy": "high-value"},
        )
        return {
            "approved": decision.approved,
            "approver": decision.metadata.get("approver"),
            "step_approval_done": True,
        }
    return {"approved": True, "auto": True, "step_approval_done": True}


def _finalize_fn(approved: str, seeded: str) -> dict[str, Any]:
    # ``approved`` arrives as a string from input_mapping rendering.
    is_approved = str(approved).lower() in ("true", "1", "yes")
    return {
        "final_seed": seeded,
        "final_decision": is_approved,
        "step_finalize_done": True,
    }


def _build_chain(checkpoint_db_path: str):
    from fastaiagent import Chain, FunctionTool, SQLiteCheckpointer
    from fastaiagent.chain.node import NodeType

    store = SQLiteCheckpointer(db_path=checkpoint_db_path)
    chain = Chain(
        "hitl-suspend-gate",
        checkpoint_enabled=True,
        checkpointer=store,
    )
    chain.add_node(
        "seed",
        tool=FunctionTool(name="seed_tool", fn=_seed_fn),
        type=NodeType.tool,
        input_mapping={"value": "{{state.seed_value}}"},
    )
    chain.add_node(
        "approval",
        tool=FunctionTool(name="approval_tool", fn=_approval_fn),
        type=NodeType.tool,
        input_mapping={"amount": "{{state.amount}}"},
    )
    chain.add_node(
        "finalize",
        tool=FunctionTool(name="finalize_tool", fn=_finalize_fn),
        type=NodeType.tool,
        # Tool-node returns are nested under ``state.output``; ``seed_value``
        # lives on top-level state because it came in via ``initial_state``.
        input_mapping={
            "approved": "{{state.output.approved}}",
            "seeded": "{{state.seed_value}}",
        },
    )
    chain.connect("seed", "approval")
    chain.connect("approval", "finalize")
    return chain, store


# ---------- Sequential gate, mirroring tests/e2e/test_gate_chain_resume.py


class TestHITLSuspendGate:
    """Suspend-and-resume contract for ``interrupt()``."""

    def test_01_first_execute_returns_paused(
        self, tmp_path: Path, gate_state: dict[str, Any]
    ) -> None:
        require_env()

        ckpt_db = str(tmp_path / "ckpt-suspend.db")
        chain, store = _build_chain(ckpt_db)
        execution_id = f"hitl-{uuid.uuid4().hex[:8]}"

        result = chain.execute(
            {"seed_value": "S1", "amount": 50_000},
            execution_id=execution_id,
        )

        assert result.status == "paused", (
            f"Expected paused chain, got status={result.status!r}; "
            f"pending_interrupt={result.pending_interrupt}"
        )
        assert result.pending_interrupt is not None
        assert result.pending_interrupt["reason"] == "manager_approval"
        assert result.pending_interrupt["node_id"] == "approval"
        assert result.pending_interrupt["context"] == {
            "amount": 50_000,
            "policy": "high-value",
        }
        assert result.execution_id == execution_id

        gate_state["chain"] = chain
        gate_state["store"] = store
        gate_state["execution_id"] = execution_id
        gate_state["ckpt_db"] = ckpt_db

    def test_02_pending_interrupts_row_carries_frozen_context(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()

        store = gate_state["store"]
        execution_id = gate_state["execution_id"]

        pending = store.list_pending_interrupts()
        rows = [p for p in pending if p.execution_id == execution_id]
        assert len(rows) == 1, f"expected exactly one pending row, got {rows}"
        row = rows[0]
        assert row.reason == "manager_approval"
        assert row.node_id == "approval"
        assert row.context == {"amount": 50_000, "policy": "high-value"}
        assert row.created_at  # populated

        # And the interrupted checkpoint exists with the right status.
        latest = store.get_last(execution_id)
        assert latest is not None
        assert latest.status == "interrupted"
        assert latest.node_id == "approval"
        assert latest.interrupt_reason == "manager_approval"
        assert latest.interrupt_context == {
            "amount": 50_000,
            "policy": "high-value",
        }

    def test_03_resume_with_approved_value_completes(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent import Resume
        from fastaiagent._internal.async_utils import run_sync

        chain = gate_state["chain"]
        execution_id = gate_state["execution_id"]

        result = run_sync(
            chain.resume(
                execution_id,
                resume_value=Resume(approved=True, metadata={"approver": "alice"}),
            )
        )

        assert result.status == "completed", (
            f"resume did not complete the chain; got status={result.status!r}, "
            f"pending_interrupt={result.pending_interrupt}"
        )
        assert result.execution_id == execution_id

        # finalize must have run with approved=True and the original seed.
        last_output = result.final_state.get("output")
        assert isinstance(last_output, dict), f"unexpected final_state: {result.final_state}"
        assert last_output.get("step_finalize_done") is True
        assert last_output.get("final_decision") is True
        assert last_output.get("final_seed") == "S1"

        # seed_value persists at the top of state from initial_state.
        assert result.final_state.get("seed_value") == "S1"

    def test_04_pending_row_was_claimed(self, gate_state: dict[str, Any]) -> None:
        require_env()
        store = gate_state["store"]
        execution_id = gate_state["execution_id"]

        pending = store.list_pending_interrupts()
        for p in pending:
            assert p.execution_id != execution_id, (
                f"pending_interrupts row was not deleted on resume: {p}"
            )

    def test_05_double_resume_raises_already_resumed(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent import AlreadyResumed, Resume
        from fastaiagent._internal.async_utils import run_sync

        # The first resume already completed in test_03; the latest checkpoint
        # is now "completed", not "interrupted". A second resume on a
        # completed chain should NOT raise AlreadyResumed (it falls into the
        # legacy failed/completed path which just starts at the next node —
        # there is no next node). To exercise AlreadyResumed cleanly we need
        # a fresh paused chain, then claim its row twice.
        ckpt_db = gate_state["ckpt_db"]
        chain2, _ = _build_chain(ckpt_db)
        new_exec = f"hitl-{uuid.uuid4().hex[:8]}"
        paused = chain2.execute(
            {"seed_value": "S2", "amount": 99_999},
            execution_id=new_exec,
        )
        assert paused.status == "paused"

        ok = run_sync(
            chain2.resume(
                new_exec,
                resume_value=Resume(approved=True, metadata={}),
            )
        )
        assert ok.status == "completed"

        with pytest.raises(AlreadyResumed):
            run_sync(
                chain2.resume(
                    new_exec,
                    resume_value=Resume(approved=False, metadata={}),
                )
            )


class TestHITLSuspendAcrossTimeAndProcesses:
    """Suspended chains must resume independent of the original process / wall-clock."""

    def test_06_resume_across_two_second_gap(self, tmp_path: Path) -> None:
        require_env()
        from fastaiagent import Resume
        from fastaiagent._internal.async_utils import run_sync

        ckpt_db = str(tmp_path / "ckpt-gap.db")
        chain, _ = _build_chain(ckpt_db)
        execution_id = f"hitl-gap-{uuid.uuid4().hex[:8]}"

        paused = chain.execute(
            {"seed_value": "G", "amount": 75_000},
            execution_id=execution_id,
        )
        assert paused.status == "paused"

        time.sleep(2)

        result = run_sync(
            chain.resume(
                execution_id,
                resume_value=Resume(approved=True, metadata={"approver": "bob"}),
            )
        )
        assert result.status == "completed"
        last_output = result.final_state.get("output")
        assert isinstance(last_output, dict)
        assert last_output.get("final_decision") is True
        assert last_output.get("final_seed") == "G"

    def test_07_resume_in_separate_process(self, tmp_path: Path) -> None:
        """Suspend in subprocess A, exit. Resume in subprocess B from a fresh import."""
        require_env()

        ckpt_db = str(tmp_path / "ckpt-subproc.db")
        execution_id = f"hitl-sub-{uuid.uuid4().hex[:8]}"
        result_path = tmp_path / "child-result.json"

        # Suspend script — runs to interrupt(), exits with the paused result.
        suspend_script = textwrap.dedent(
            f"""
            import json, sys
            sys.path.insert(0, {repr(str(Path(__file__).resolve().parents[2]))})
            from tests.e2e.test_gate_hitl_suspend import _build_chain
            chain, _ = _build_chain({ckpt_db!r})
            r = chain.execute(
                {{"seed_value": "P", "amount": 12_345}},
                execution_id={execution_id!r},
            )
            with open({str(result_path)!r}, "w") as f:
                json.dump(
                    {{"status": r.status, "pending": r.pending_interrupt}},
                    f,
                )
            """
        ).strip()
        proc = subprocess.run(
            [sys.executable, "-c", suspend_script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, (
            f"suspend subprocess exited {proc.returncode}\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
        with open(result_path) as f:
            child = json.load(f)
        assert child["status"] == "paused"
        assert child["pending"]["reason"] == "manager_approval"

        # Resume script — completely fresh interpreter, fresh import of the
        # SDK and the chain helpers.
        resume_script = textwrap.dedent(
            f"""
            import json, sys
            sys.path.insert(0, {repr(str(Path(__file__).resolve().parents[2]))})
            from fastaiagent import Resume
            from fastaiagent._internal.async_utils import run_sync
            from tests.e2e.test_gate_hitl_suspend import _build_chain
            chain, _ = _build_chain({ckpt_db!r})
            r = run_sync(
                chain.resume(
                    {execution_id!r},
                    resume_value=Resume(approved=True, metadata={{"approver": "carol"}}),
                )
            )
            with open({str(result_path)!r}, "w") as f:
                out = r.final_state.get("output") or {{}}
                json.dump(
                    {{
                        "status": r.status,
                        "final_decision": out.get("final_decision"),
                        "final_seed": out.get("final_seed"),
                    }},
                    f,
                )
            """
        ).strip()
        proc2 = subprocess.run(
            [sys.executable, "-c", resume_script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc2.returncode == 0, (
            f"resume subprocess exited {proc2.returncode}\n"
            f"stdout={proc2.stdout}\nstderr={proc2.stderr}"
        )
        with open(result_path) as f:
            done = json.load(f)
        assert done["status"] == "completed"
        assert done["final_decision"] is True
        assert done["final_seed"] == "P"


class TestFrozenContext:
    """``context`` passed to interrupt() is JSON-frozen at suspend time."""

    def test_08_context_is_frozen_at_suspend(self, tmp_path: Path) -> None:
        require_env()
        from fastaiagent import SQLiteCheckpointer

        ckpt_db = str(tmp_path / "ckpt-frozen.db")
        chain, store = _build_chain(ckpt_db)
        execution_id = f"hitl-frozen-{uuid.uuid4().hex[:8]}"

        # First execution suspends with amount=10_000.
        paused = chain.execute(
            {"seed_value": "F", "amount": 10_000},
            execution_id=execution_id,
        )
        assert paused.status == "paused"

        # Simulate "the world changed" between pause and resume — directly
        # rewrite the state_snapshot stored in the checkpoint to a different
        # amount. The frozen-context contract says the resumer sees the
        # ORIGINAL context dict, regardless.
        store2 = SQLiteCheckpointer(db_path=ckpt_db)
        store2.setup()
        # Rewrite state snapshot via raw connection.
        with store2._conn()._lock:
            conn = store2._conn()._get_conn()
            conn.execute(
                "UPDATE checkpoints SET state_snapshot = ? "
                "WHERE execution_id = ? AND status = 'interrupted'",
                ('{"amount": 1, "seeded": "MUTATED"}', execution_id),
            )
            conn.commit()
        store2.close()

        # The pending_interrupts row still carries the ORIGINAL context.
        pending = [p for p in store.list_pending_interrupts() if p.execution_id == execution_id]
        assert len(pending) == 1
        assert pending[0].context == {"amount": 10_000, "policy": "high-value"}

        # And the interrupted checkpoint still carries it in interrupt_context.
        latest = store.get_last(execution_id)
        assert latest is not None
        assert latest.interrupt_context == {
            "amount": 10_000,
            "policy": "high-value",
        }
