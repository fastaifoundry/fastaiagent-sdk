"""End-to-end quality gate — crash recovery (spec test #1).

Scenario: a 5-node linear chain. A subprocess executes it. step_3 sleeps long
enough that the parent test can ``SIGKILL`` the subprocess mid-sleep. Once
the subprocess is dead, the parent reads the local SQLite checkpointer
directly to confirm the last successful checkpoint is step_2, then calls
``chain.resume(execution_id)`` *in-process* and verifies that:

    1. resume starts at step_3 (the node where the crash happened)
    2. nodes 3, 4, 5 all complete
    3. the post-resume node_results contain exactly steps 3-5 (the parent
       didn't re-run 1 or 2)
    4. the checkpoint table ends with one row per node (5 total) — two
       written by the dead worker, three by the resumer

This is the marketing claim ("crash-proof agents") test. Run multiple
iterations to catch flakiness; CI bumps the loop count via
``E2E_CRASH_LOOPS=10``.
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


# ---------- Chain-builder helper used by parent + subprocess --------------


def _step_fn(idx: int) -> Any:
    """Factory: each step returns a unique-completion marker."""

    def fn() -> dict[str, Any]:
        return {f"step{idx}_done": True, "step": idx}

    fn.__name__ = f"step_{idx}_fn"
    return fn


def _step_3_slow() -> dict[str, Any]:
    """The crash zone — long enough for the parent to SIGKILL mid-execution."""
    time.sleep(5)
    return {"step3_done": True, "step": 3}


def _build_chain(checkpoint_db_path: str) -> tuple[Any, Any]:
    """Build the same 5-node chain in the worker and the parent."""
    from fastaiagent import Chain, FunctionTool, SQLiteCheckpointer
    from fastaiagent.chain.node import NodeType

    fns = [_step_fn(1), _step_fn(2), _step_3_slow, _step_fn(4), _step_fn(5)]

    store = SQLiteCheckpointer(db_path=checkpoint_db_path)
    chain = Chain("crash-recovery-gate", checkpoint_enabled=True, checkpointer=store)
    for i, fn in enumerate(fns, start=1):
        chain.add_node(
            f"step_{i}",
            tool=FunctionTool(name=f"step_{i}_tool", fn=fn),
            type=NodeType.tool,
        )
    for i in range(1, 5):
        chain.connect(f"step_{i}", f"step_{i + 1}")
    return chain, store


# ---------- Subprocess driver ---------------------------------------------


def _spawn_worker(ckpt_db: str, execution_id: str) -> subprocess.Popen[bytes]:
    """Run the chain in a subprocess via a fresh-import script."""
    repo_root = str(Path(__file__).resolve().parents[2])
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {repo_root!r})
        from tests.e2e.test_gate_crash_recovery import _build_chain
        chain, _store = _build_chain({ckpt_db!r})
        # This call will be killed mid-step_3 by the parent. The chain will
        # checkpoint step_1 and step_2 before that happens.
        chain.execute({{}}, execution_id={execution_id!r})
        """
    ).strip()
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for_step_2_checkpoint(ckpt_db: str, execution_id: str, timeout: float = 10.0) -> None:
    """Poll the checkpoint store until step_2 is committed.

    We use a fresh ``SQLiteCheckpointer`` per poll so we read the worker's
    latest committed state via SQLite WAL. Returns once ``step_2`` is the
    latest checkpoint, or raises ``TimeoutError``.
    """
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
        if latest is not None and latest.node_id == "step_2":
            return
        time.sleep(0.05)
    raise TimeoutError(
        f"step_2 was never checkpointed within {timeout}s for execution {execution_id!r}"
    )


def _kill_and_reap(proc: subprocess.Popen[bytes], timeout: float = 10.0) -> None:
    """SIGKILL the worker and wait for the process to be reaped.

    SIGKILL (not SIGTERM) — we want to simulate a real crash, not a clean
    shutdown. The chain executor has no opportunity to flush extra state.
    """
    try:
        os.kill(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # already dead
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


# ---------- The crash-recovery gate ---------------------------------------


_CRASH_LOOPS = int(os.environ.get("E2E_CRASH_LOOPS", "3"))


class TestCrashRecoveryGate:
    """SIGKILL during step_3 → resume in fresh process completes the chain."""

    @pytest.mark.parametrize("run", range(_CRASH_LOOPS))
    def test_subprocess_kill_resumes_at_step_3_and_completes(
        self, tmp_path: Path, run: int
    ) -> None:
        require_env()
        from fastaiagent import SQLiteCheckpointer
        from fastaiagent._internal.async_utils import run_sync

        ckpt_db = str(tmp_path / f"ckpt-crash-{run}.db")
        execution_id = f"crash-{uuid.uuid4().hex[:8]}"

        # 1. Spawn the worker. It will checkpoint step_1, step_2, then sleep
        # 5s in step_3.
        proc = _spawn_worker(ckpt_db, execution_id)
        try:
            _wait_for_step_2_checkpoint(ckpt_db, execution_id)
            # 2. SIGKILL mid-step_3.
            _kill_and_reap(proc)
            assert proc.returncode is not None
            # SIGKILL → returncode is the negative of the signal number.
            assert proc.returncode == -signal.SIGKILL or proc.returncode < 0, (
                f"worker did not die on SIGKILL: returncode={proc.returncode}, "
                f"stdout={proc.stdout.read() if proc.stdout else b''!r}, "
                f"stderr={proc.stderr.read() if proc.stderr else b''!r}"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

        # 3. Pre-resume verification: only step_1 and step_2 are checkpointed.
        store = SQLiteCheckpointer(db_path=ckpt_db)
        store.setup()
        before = store.list(execution_id)
        before_ids = [cp.node_id for cp in before]
        assert before_ids == ["step_1", "step_2"], (
            f"pre-resume checkpoints: expected ['step_1', 'step_2'], got {before_ids}"
        )
        latest_before = store.get_last(execution_id)
        assert latest_before is not None and latest_before.node_id == "step_2"
        store.close()

        # 4. In-process resume — same DB, brand-new Chain object.
        chain, _ = _build_chain(ckpt_db)
        result = run_sync(chain.resume(execution_id))

        # 5. Resume completed cleanly.
        assert result.status == "completed", (
            f"resume did not complete: status={result.status}, "
            f"pending_interrupt={result.pending_interrupt}"
        )
        assert result.execution_id == execution_id

        # 6. node_results contain ONLY the post-resume nodes (3, 4, 5). The
        # parent did not re-run step_1 or step_2.
        assert set(result.node_results.keys()) == {"step_3", "step_4", "step_5"}, (
            f"resume re-ran the wrong nodes: {sorted(result.node_results.keys())}"
        )
        for i in (3, 4, 5):
            entry = result.node_results[f"step_{i}"]
            assert isinstance(entry, dict)
            inner = entry.get("output")
            assert isinstance(inner, dict), f"step_{i} output: {entry!r}"
            assert inner.get(f"step{i}_done") is True
            assert inner.get("step") == i

        # 7. The checkpoint table ends with all five nodes. step_3..5 were
        # written by the in-process resume; step_1..2 by the dead worker.
        store2 = SQLiteCheckpointer(db_path=ckpt_db)
        store2.setup()
        after = store2.list(execution_id)
        after_ids = [cp.node_id for cp in after]
        assert after_ids == ["step_1", "step_2", "step_3", "step_4", "step_5"], (
            f"final checkpoint chain: {after_ids}"
        )
        latest_after = store2.get_last(execution_id)
        assert latest_after is not None and latest_after.node_id == "step_5"
        # The post-resume checkpoints' created_at must come AFTER the
        # pre-resume ones — proves the resume actually ran the nodes rather
        # than reading them from somewhere stale.
        worker_ids = ("step_1", "step_2")
        resume_ids = ("step_3", "step_4", "step_5")
        worker_ts = max(cp.created_at for cp in after if cp.node_id in worker_ids)
        resume_ts = min(cp.created_at for cp in after if cp.node_id in resume_ids)
        assert resume_ts > worker_ts, (
            f"resume checkpoints ({resume_ts}) should be newer than worker "
            f"checkpoints ({worker_ts})"
        )
        store2.close()
