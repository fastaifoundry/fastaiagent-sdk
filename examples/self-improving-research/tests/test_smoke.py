"""Smoke tests for examples/self-improving-research — no live LLM calls.

These verify the example's imports work and the wiring between the
deep-research-agent template, ``PersistentFactBlock``, and the new learn
module is intact. No phase actually runs end-to-end here — that's the
job of the integration test in ``tests/integration/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Also expose deep-research-agent so the import inside agent.py resolves.
_DEEP = Path(__file__).resolve().parent.parent.parent / "deep-research-agent"
if str(_DEEP) not in sys.path:
    sys.path.insert(0, str(_DEEP))


def test_imports() -> None:
    import agent  # noqa: F401


def test_memory_setup_returns_composable_memory(tmp_path, monkeypatch) -> None:
    """The deep-research-agent's memory_setup should now return a real memory."""
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    monkeypatch.delenv("DEEP_RESEARCH_DISABLE_LEARNED_MEMORY", raising=False)

    import importlib

    import memory_setup as ms

    importlib.reload(ms)
    mem = ms.build_memory()
    assert mem is not None, "PR B should activate memory in the deep-research template"


def test_disable_flag_short_circuits(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    monkeypatch.setenv("DEEP_RESEARCH_DISABLE_LEARNED_MEMORY", "1")

    import importlib

    import memory_setup as ms

    importlib.reload(ms)
    assert ms.build_memory() is None


def test_self_test_flag_runs_clean(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    import importlib

    import agent as agent_mod

    importlib.reload(agent_mod)
    monkeypatch.setattr("sys.argv", ["agent.py", "--self-test"])
    agent_mod.main()  # should print "self-test: ok" and return cleanly
