"""End-to-end quality gate — Chain.resume() from a checkpoint.

The happy-path Chain gate (test_gate_chain.py) only proves that a chain
runs cleanly from start to finish. This gate proves the *failure-and-
resume* contract: when a chain crashes mid-flight, the state from
already-completed nodes is checkpointed, and ``Chain.resume(execution_id)``
picks up at the right node and runs to completion. With ``modified_state``,
the resume can also patch values to fix whatever caused the original
failure.

Scenario (3 tool nodes wired linearly):
    step_a -> step_b (flaky) -> step_c

- ``seed_value`` is seeded into chain state via the initial_state to
  ``chain.execute(...)``. This is important because tool nodes have a
  quirk where each node's return value gets wrapped in
  ``{"output": ..., "error": ...}`` and merged into state, so each
  successive node overwrites ``state.output`` with its own output.
  The only reliable way to thread a value across multiple tool nodes
  is to put it on the top-level state (via initial_state, or via
  modified_state on resume), where nothing overwrites it.
- step_a always succeeds. Its existence is what creates the checkpoint
  the resume will pick up from.
- step_b raises a RuntimeError on its FIRST invocation (per a process-
  scoped attempt counter), then succeeds on subsequent invocations.
- step_c reads ``seed_value`` from state and reports it back.

First execute() raises a ToolExecutionError (step_b's failure surfaces
as the SDK's wrapped exception). At that point the checkpoint store
contains exactly one checkpoint — for step_a. step_b never completed,
so no checkpoint for it. step_c was never reached.

Then chain.resume(execution_id, modified_state={"seed_value": "patched"})
should:
  1. Load the latest checkpoint (step_a),
  2. Apply ``modified_state`` on top of the snapshot — overriding
     ``seed_value`` from "original" to "patched",
  3. Identify step_b as the next node,
  4. Re-execute step_b — which now succeeds (counter > 0) and sees
     the patched value via its input_mapping,
  5. Continue through step_c, which also sees the patched value,
  6. Return a ChainResult whose final_state.seed_value == "patched"
     and whose latest output (step_c) reflects the patched value.

The gate is fully self-contained: no LLM, no network, no platform.
Everything runs deterministically against a temp-dir SQLite checkpoint
store.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


# Per-test attempt counter for the flaky tool. Reset by each test method
# via a fresh dict so test_03 (positive control) starts from zero.
_FLAKY_STATE: dict[str, int] = {"attempts": 0}


def _reset_flaky() -> None:
    _FLAKY_STATE["attempts"] = 0


def _step_a_fn() -> dict[str, Any]:
    """Always succeeds. Its purpose is to be the checkpointed node before
    the failure, so resume() has something to pick up from.

    Note: any keys in this dict end up nested under ``state.output`` in
    the chain state because of how the chain executor wraps tool node
    results. We don't try to use that for cross-node value passing —
    seed_value lives on top-level state via initial_state instead.
    """
    return {"step_a_done": True}


def _flaky_step_b_fn(seed: str) -> dict[str, Any]:
    """Fails on first call, succeeds on subsequent calls.

    Used to simulate a transient failure that resume() can recover from.
    """
    _FLAKY_STATE["attempts"] += 1
    if _FLAKY_STATE["attempts"] == 1:
        raise RuntimeError(f"simulated transient failure on attempt {_FLAKY_STATE['attempts']}")
    return {
        "step_b_processed": seed,
        "step_b_attempt": _FLAKY_STATE["attempts"],
        "step_b_done": True,
    }


def _step_c_fn(seed: str) -> dict[str, Any]:
    """Reads seed_value from chain state via input_mapping and reports it."""
    return {"final_seed": seed, "step_c_done": True}


def _build_chain(checkpoint_db_path: str):
    """Construct the linear test chain with an isolated checkpoint store."""
    from fastaiagent import Chain, FunctionTool, SQLiteCheckpointer
    from fastaiagent.chain.node import NodeType

    store = SQLiteCheckpointer(db_path=checkpoint_db_path)
    chain = Chain(
        "chain-resume-gate",
        checkpoint_enabled=True,
        checkpointer=store,
    )
    chain.add_node(
        "step_a",
        tool=FunctionTool(name="step_a_tool", fn=_step_a_fn),
        type=NodeType.tool,
    )
    chain.add_node(
        "step_b",
        tool=FunctionTool(name="flaky_step_b_tool", fn=_flaky_step_b_fn),
        type=NodeType.tool,
        input_mapping={"seed": "{{state.seed_value}}"},
    )
    chain.add_node(
        "step_c",
        tool=FunctionTool(name="step_c_tool", fn=_step_c_fn),
        type=NodeType.tool,
        input_mapping={"seed": "{{state.seed_value}}"},
    )
    chain.connect("step_a", "step_b")
    chain.connect("step_b", "step_c")
    return chain, store


class TestChainResumeGate:
    """Failure → checkpoint → resume → completion contract for Chain."""

    def test_01_first_execute_fails_at_step_b(
        self, tmp_path: Path, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        from fastaiagent._internal.errors import ToolExecutionError

        _reset_flaky()
        ckpt_db = str(tmp_path / "ckpt-resume.db")
        chain, store = _build_chain(ckpt_db)
        execution_id = f"resume-gate-{uuid.uuid4().hex[:8]}"

        with pytest.raises(ToolExecutionError):
            chain.execute(
                {"seed_value": "original", "message": "kick off"},
                execution_id=execution_id,
            )

        # Stash state for the next sub-tests in this class.
        gate_state["chain"] = chain
        gate_state["store"] = store
        gate_state["execution_id"] = execution_id
        gate_state["ckpt_db"] = ckpt_db

    def test_02_checkpoint_contains_step_a_only(self, gate_state: dict[str, Any]) -> None:
        require_env()
        store = gate_state["store"]
        execution_id = gate_state["execution_id"]

        checkpoints = store.list(execution_id)
        node_ids = [cp.node_id for cp in checkpoints]
        assert "step_a" in node_ids, (
            f"step_a checkpoint missing — checkpoint store did not record "
            f"the successful node before the failure: {node_ids}"
        )
        assert "step_b" not in node_ids, (
            f"step_b checkpoint should not exist — the node raised before "
            f"completing, so it was never checkpointed: {node_ids}"
        )
        assert "step_c" not in node_ids, (
            f"step_c was never reached, must not have a checkpoint: {node_ids}"
        )

        latest = store.get_last(execution_id)
        assert latest is not None
        assert latest.node_id == "step_a"
        assert latest.state_snapshot.get("seed_value") == "original", (
            f"step_a snapshot did not capture seed_value: {latest.state_snapshot}"
        )

    def test_03_resume_with_modified_state_completes(self, gate_state: dict[str, Any]) -> None:
        """Resume should re-run step_b (now succeeding) and finish step_c.

        The modified_state override should patch seed_value before
        execution continues, so step_b and step_c both see "patched"
        instead of "original".
        """
        require_env()
        from fastaiagent._internal.async_utils import run_sync

        chain = gate_state["chain"]
        execution_id = gate_state["execution_id"]

        # _FLAKY_STATE["attempts"] is currently 1 (from the failed test_01
        # invocation). The next call will succeed.
        result = run_sync(
            chain.resume(
                execution_id,
                modified_state={"seed_value": "patched"},
            )
        )

        assert result is not None, "resume returned None"
        assert result.execution_id == execution_id, "resumed run lost the original execution_id"
        # The flaky tool was retried and produced output on the second call.
        assert _FLAKY_STATE["attempts"] >= 2, (
            f"flaky tool was not retried; attempts={_FLAKY_STATE['attempts']}"
        )

        # final_state should reflect both the carried-over data from
        # step_a's checkpoint AND the modified_state override.
        # seed_value lives at the top level (it was in initial_state and
        # nothing overwrites it), so it's preserved with the patched value.
        final_state = result.final_state
        assert final_state.get("seed_value") == "patched", (
            f"modified_state did not override seed_value on resume: {final_state}"
        )

        # Per the chain executor's tool-node wrapping behavior, each tool
        # node's return dict ends up nested under state.output, and the
        # NEXT tool node overwrites that key. So state.output reflects the
        # LAST step that ran — step_c. Verify it ran and saw the patched
        # seed value via its input_mapping.
        last_output = final_state.get("output")
        assert isinstance(last_output, dict), (
            f"state.output should hold the last step's return dict, got: {last_output!r}"
        )
        assert last_output.get("step_c_done") is True, (
            f"step_c did not complete on resume — state.output is not from step_c: {last_output}"
        )
        assert last_output.get("final_seed") == "patched", (
            f"step_c did not see the patched seed_value: {last_output}"
        )

    def test_04_resumed_run_recheckpointed_steps_b_and_c(self, gate_state: dict[str, Any]) -> None:
        """After a successful resume, step_b and step_c should be checkpointed."""
        require_env()
        store = gate_state["store"]
        execution_id = gate_state["execution_id"]

        checkpoints = store.list(execution_id)
        node_ids = [cp.node_id for cp in checkpoints]
        assert "step_b" in node_ids, (
            f"step_b checkpoint missing after successful resume: {node_ids}"
        )
        assert "step_c" in node_ids, (
            f"step_c checkpoint missing after successful resume: {node_ids}"
        )

        # Latest checkpoint should be step_c (the last node).
        latest = store.get_last(execution_id)
        assert latest is not None
        assert latest.node_id == "step_c", f"latest checkpoint is not step_c: {latest.node_id}"
        # Top-level seed_value carried through the resume's modified_state.
        assert latest.state_snapshot.get("seed_value") == "patched", (
            f"latest checkpoint missing patched seed_value: {latest.state_snapshot}"
        )
        # And step_c's nested output reflects the patched seed.
        last_out = latest.state_snapshot.get("output")
        assert isinstance(last_out, dict), (
            f"latest checkpoint's state.output is not a dict: {last_out!r}"
        )
        assert last_out.get("final_seed") == "patched", (
            f"step_c output in checkpoint did not see patched seed: {last_out}"
        )

    def test_05_resume_unknown_execution_id_raises(
        self, tmp_path: Path, gate_state: dict[str, Any]
    ) -> None:
        """Calling resume() with an unknown execution_id raises ChainCheckpointError."""
        require_env()
        from fastaiagent._internal.async_utils import run_sync
        from fastaiagent._internal.errors import ChainCheckpointError

        _reset_flaky()
        ckpt_db = str(tmp_path / "ckpt-resume-empty.db")
        chain, _store = _build_chain(ckpt_db)

        with pytest.raises(ChainCheckpointError):
            run_sync(chain.resume("nonexistent-execution-id"))
