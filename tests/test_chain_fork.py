"""Chain.afork — checkpoint-based fork into a new, independent execution.

No mocks, no LLM: deterministic tool nodes over a real SQLite checkpointer.
Verifies the 2.4a exit gate — resume-from-checkpoint reproduces the forward
state; fork-from-step with a modified state diverges; the ORIGINAL execution is
left completely intact.
"""

from __future__ import annotations

from pathlib import Path

from fastaiagent import Chain, FunctionTool
from fastaiagent.chain.node import NodeType
from fastaiagent.checkpointers import SQLiteCheckpointer


def _double(x: str) -> int:
    return int(x) * 2


def _addk(x: str, k: str) -> int:
    return int(x) + int(k)


def _build_chain(db_path: str):
    cp = SQLiteCheckpointer(db_path=db_path)
    chain = Chain("fork-test", checkpoint_enabled=True, checkpointer=cp)
    chain.add_node(
        "n1",
        tool=FunctionTool(name="double", fn=_double),
        type=NodeType.tool,
        input_mapping={"x": "{{state.start}}"},
    )
    chain.add_node(
        "n2",
        tool=FunctionTool(name="addk", fn=_addk),
        type=NodeType.tool,
        input_mapping={"x": "{{state.output}}", "k": "{{state.k}}"},
    )
    chain.connect("n1", "n2")
    return chain, cp


class TestChainFork:
    def test_fork_reproduces_diverges_and_leaves_original_intact(self, tmp_path: Path):
        db = str(tmp_path / "ckpt.db")
        chain, cp = _build_chain(db)

        # Original run: n1 double(5)=10 -> output=10 ; n2 addk(10,10)=20 -> output=20
        orig = chain.execute({"start": 5, "k": 10}, execution_id="orig")
        assert orig.execution_id == "orig"
        assert orig.final_state["output"] == 20

        ckpts = cp.list("orig")
        n1 = next(c for c in ckpts if c.node_id == "n1")

        # (a) RESUME/REPRODUCE: fork from n1 with no change -> same forward
        #     result, but a fresh execution_id (original untouched).
        repro = chain.fork("orig", checkpoint_id=n1.checkpoint_id)
        assert repro.execution_id != "orig"
        assert repro.final_state["output"] == 20

        # (b) DIVERGE: fork from n1 with a modified state -> n2 sees 100, not 10.
        div = chain.fork("orig", checkpoint_id=n1.checkpoint_id, modified_state={"output": 100})
        assert div.execution_id != "orig"
        assert div.final_state["output"] == 110  # addk(100, 10)
        assert div.final_state["output"] != orig.final_state["output"]

        # (c) ORIGINAL INTACT: the source execution's checkpoints are unchanged.
        orig_after = cp.list("orig")
        # ``run_end`` is the terminal marker the original run wrote when it
        # finished (audit D5) — a fork must not disturb it either.
        assert {c.node_id for c in orig_after} == {"n1", "n2", "run_end"}
        n2_orig = next(c for c in orig_after if c.node_id == "n2")
        assert n2_orig.state_snapshot["output"] == 20

        # (d) LINEAGE: the fork's origin checkpoint links back to the source step.
        fork_ckpts = cp.list(div.execution_id)
        origin = next(c for c in fork_ckpts if c.node_id == "__fork_origin__")
        assert origin.parent_checkpoint_id == n1.checkpoint_id

    def test_fork_from_final_node_is_rejected(self, tmp_path: Path):
        """A completed linear chain has nothing downstream of its last node.

        This test passed before 1.67.0 for the wrong reason, and its comment
        said so: it claimed the last checkpoint was ``n2``. Since 1.65.0 the
        last checkpoint is the ``run_end`` marker, so the refusal fired on a
        *tombstone* and its message named ``run_end`` as "the final node" —
        which it is not, being no node at all. The fix must keep the refusal and
        make it truthful, so the message is asserted, not just the type.
        """
        import pytest

        from fastaiagent._internal.errors import ChainResumeError

        db = str(tmp_path / "ckpt2.db")
        chain, cp = _build_chain(db)
        chain.execute({"start": 1, "k": 1}, execution_id="orig2")

        # The newest row really is the marker — that is the thing fork must skip.
        assert cp.get_last("orig2").node_id == "run_end"

        with pytest.raises(ChainResumeError) as excinfo:
            chain.fork("orig2")  # defaults to the last EXECUTED step (n2)
        message = str(excinfo.value)
        assert "run_end" not in message, (
            "the refusal named the run-end marker as a node; it is a tombstone"
        )
        assert "'n2'" in message
        # And it must point at the way out, which is an explicit earlier step.
        assert "checkpoint_id" in message

    def test_forking_a_failed_run_branches_from_its_last_real_step(self, tmp_path: Path):
        """The one true regression of this whole programme.

        Before 1.65.0 this worked: fork took ``get_last``, found ``n1``, and ran
        forward. The run-end marker then became the newest row on every finished
        run — including failed ones — and both fork paths were still calling
        ``get_last`` raw, so this raised ``ChainResumeError`` naming ``run_end``.
        """
        import pytest

        state = {"explode": True}

        def _maybe_boom(x: str) -> int:
            if state["explode"]:
                raise RuntimeError("n2 blew up")
            return int(x) + 1

        db = str(tmp_path / "failed.db")
        cp = SQLiteCheckpointer(db_path=db)
        chain = Chain("fork-failed", checkpoint_enabled=True, checkpointer=cp)
        chain.add_node(
            "n1",
            tool=FunctionTool(name="double", fn=_double),
            type=NodeType.tool,
            input_mapping={"x": "{{state.start}}"},
        )
        chain.add_node(
            "n2",
            tool=FunctionTool(name="boom", fn=_maybe_boom),
            type=NodeType.tool,
            input_mapping={"x": "{{state.output}}"},
        )
        chain.connect("n1", "n2")

        with pytest.raises(Exception):
            chain.execute({"start": 5}, execution_id="failed")

        rows = cp.list("failed")
        assert [c.node_id for c in rows] == ["n1", "run_end"]
        assert rows[-1].status == "failed"

        # Repair the cause, then branch the dead run forward from where it got to.
        state["explode"] = False
        branch = chain.fork("failed")

        assert branch.execution_id != "failed"
        assert branch.status == "completed"
        assert branch.final_state["output"] == 11  # n2 over n1's 10
        # It branched from n1's real checkpoint, not from the tombstone, and the
        # branch's state never inherited the marker's ``run_status``.
        origin = next(c for c in cp.list(branch.execution_id) if c.node_id == "__fork_origin__")
        n1_row = next(c for c in rows if c.node_id == "n1")
        assert origin.parent_checkpoint_id == n1_row.checkpoint_id
        assert "run_status" not in origin.state_snapshot
        assert "run_error" not in origin.state_snapshot

    def test_forking_a_run_that_holds_only_its_marker_is_a_checkpoint_error(self, tmp_path: Path):
        """A run that died at node 0 has no step to branch from, and must say so.

        The failure mode matters: it used to be a ``ChainResumeError`` claiming
        ``run_end`` was the final node of the chain. The truthful answer is the
        same one an unknown execution_id gets — there is no checkpoint to fork.
        """
        import pytest

        from fastaiagent._internal.errors import ChainCheckpointError

        def _boom(x: str) -> int:
            raise RuntimeError("died at node 0")

        cp = SQLiteCheckpointer(db_path=str(tmp_path / "dead0.db"))
        chain = Chain("fork-dead0", checkpoint_enabled=True, checkpointer=cp)
        chain.add_node(
            "n1",
            tool=FunctionTool(name="boom", fn=_boom),
            type=NodeType.tool,
            input_mapping={"x": "{{state.start}}"},
        )
        chain.add_node(
            "n2",
            tool=FunctionTool(name="double", fn=_double),
            type=NodeType.tool,
            input_mapping={"x": "{{state.output}}"},
        )
        chain.connect("n1", "n2")

        with pytest.raises(Exception):
            chain.execute({"start": 5}, execution_id="dead0")
        assert [c.node_id for c in cp.list("dead0")] == ["run_end"]

        with pytest.raises(ChainCheckpointError) as excinfo:
            chain.fork("dead0")
        assert "No checkpoint found to fork" in str(excinfo.value)

    def test_forking_the_marker_by_id_says_what_it_is(self, tmp_path: Path):
        """Naming the tombstone explicitly is a caller mistake, not an empty
        history, so it gets its own message rather than "nothing to fork"."""
        import pytest

        from fastaiagent._internal.errors import ChainCheckpointError

        db = str(tmp_path / "byid.db")
        chain, cp = _build_chain(db)
        chain.execute({"start": 1, "k": 1}, execution_id="byid")
        marker = next(c for c in cp.list("byid") if c.node_id == "run_end")

        with pytest.raises(ChainCheckpointError) as excinfo:
            chain.fork("byid", checkpoint_id=marker.checkpoint_id)
        assert "run-end marker" in str(excinfo.value)

    def test_fork_restores_a_plane_held_run_before_deciding_there_is_nothing(
        self, tmp_path: Path, monkeypatch
    ):
        """``restore_if_missing`` was wired into all four resume paths and
        neither fork path, so forking a run this machine had never seen failed
        even when the plane was holding it.

        The restore itself is a no-op when disconnected — what is asserted here
        is that ``afork`` *calls* it, and that it calls it before reading the
        store, by having the restore populate an empty store.
        """

        # Run a chain against one store, then fork it against an EMPTY one whose
        # only way to see the run is the restore hook.
        source = SQLiteCheckpointer(db_path=str(tmp_path / "source.db"))
        producer = Chain("restorable", checkpoint_enabled=True, checkpointer=source)
        producer.add_node(
            "n1",
            tool=FunctionTool(name="double", fn=_double),
            type=NodeType.tool,
            input_mapping={"x": "{{state.start}}"},
        )
        producer.add_node(
            "n2",
            tool=FunctionTool(name="addk", fn=_addk),
            type=NodeType.tool,
            input_mapping={"x": "{{state.output}}", "k": "{{state.k}}"},
        )
        producer.connect("n1", "n2")
        producer.execute({"start": 5, "k": 10}, execution_id="held")

        fresh = SQLiteCheckpointer(db_path=str(tmp_path / "fresh.db"))
        fresh.setup()
        assert fresh.list("held") == []

        calls: list[str] = []

        def _restore(store, execution_id):
            calls.append(execution_id)
            for row in source.list(execution_id):
                store.put(row)

        import fastaiagent.checkpointers.platform_replica as replica

        monkeypatch.setattr(replica, "restore_if_missing", _restore)

        consumer = Chain("restorable", checkpoint_enabled=True, checkpointer=fresh)
        consumer.add_node(
            "n1",
            tool=FunctionTool(name="double", fn=_double),
            type=NodeType.tool,
            input_mapping={"x": "{{state.start}}"},
        )
        consumer.add_node(
            "n2",
            tool=FunctionTool(name="addk", fn=_addk),
            type=NodeType.tool,
            input_mapping={"x": "{{state.output}}", "k": "{{state.k}}"},
        )
        consumer.connect("n1", "n2")

        n1 = next(c for c in source.list("held") if c.node_id == "n1")
        branch = consumer.fork("held", checkpoint_id=n1.checkpoint_id)

        assert calls == ["held"], "afork never asked the plane for the run"
        assert branch.final_state["output"] == 20
