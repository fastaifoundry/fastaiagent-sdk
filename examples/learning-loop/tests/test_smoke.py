"""Smoke tests for examples/learning-loop — no live LLM calls.

Verifies imports, MemoryStore construction against a temp DB, and the
PersistentFactBlock-renders-cached-facts path without hitting the LLM
extractor.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``import agent`` from the example dir resolvable.
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def test_imports() -> None:
    import agent  # noqa: F401


def test_memory_store_round_trip(tmp_path, monkeypatch) -> None:
    """Insert + read against an isolated local.db."""
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))

    from fastaiagent.learn import Fact, MemoryStore

    store = MemoryStore()
    fid = store.add(
        Fact(
            scope="agent",
            scope_id="learning-loop-demo",
            fact="The demo agent introduces itself first.",
            source_trace_id="trace-1",
        )
    )
    assert fid > 0

    facts = store.list_active(scope="agent", scope_id="learning-loop-demo")
    assert len(facts) == 1
    assert facts[0].fact == "The demo agent introduces itself first."

    # Idempotency: re-add same fact → same id, no duplicate row.
    fid2 = store.add(
        Fact(
            scope="agent",
            scope_id="learning-loop-demo",
            fact="The demo agent introduces itself first.",
        )
    )
    assert fid2 == fid
    assert len(store.list_active(scope="agent", scope_id="learning-loop-demo")) == 1


def test_persistent_fact_block_renders_facts_into_prompt(
    tmp_path, monkeypatch
) -> None:
    """Verify the block reads from the store and emits a SystemMessage."""
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))

    from fastaiagent.agent.memory_blocks import PersistentFactBlock
    from fastaiagent.learn import Fact, MemoryStore

    store = MemoryStore()
    store.add(Fact(scope="agent", scope_id="x", fact="rule A applies"))
    store.add(Fact(scope="agent", scope_id="x", fact="rule B applies"))

    block = PersistentFactBlock(scope="agent", scope_id="x")
    rendered = block.render(query="anything")
    assert len(rendered) == 1
    text = rendered[0].content
    assert "rule A applies" in text
    assert "rule B applies" in text
    assert "Learned facts" in text


def test_self_test_flag_runs_clean(tmp_path, monkeypatch) -> None:
    """``python agent.py --self-test`` — used by the example test gate."""
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    import importlib

    import agent as agent_mod

    importlib.reload(agent_mod)
    monkeypatch.setattr("sys.argv", ["agent.py", "--self-test"])
    agent_mod.main()  # should print "self-test: ok" and return cleanly
