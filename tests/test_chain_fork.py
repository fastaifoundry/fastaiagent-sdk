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
        assert {c.node_id for c in orig_after} == {"n1", "n2"}
        n2_orig = next(c for c in orig_after if c.node_id == "n2")
        assert n2_orig.state_snapshot["output"] == 20

        # (d) LINEAGE: the fork's origin checkpoint links back to the source step.
        fork_ckpts = cp.list(div.execution_id)
        origin = next(c for c in fork_ckpts if c.node_id == "__fork_origin__")
        assert origin.parent_checkpoint_id == n1.checkpoint_id

    def test_fork_from_final_node_is_rejected(self, tmp_path: Path):
        import pytest

        from fastaiagent._internal.errors import ChainResumeError

        db = str(tmp_path / "ckpt2.db")
        chain, cp = _build_chain(db)
        chain.execute({"start": 1, "k": 1}, execution_id="orig2")

        # Last checkpoint is n2 (the final node) -> nothing downstream to run.
        with pytest.raises(ChainResumeError):
            chain.fork("orig2")  # defaults to last checkpoint (n2)
