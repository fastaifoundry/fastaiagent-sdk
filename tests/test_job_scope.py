"""job_scope() — per-job request-scoping of process-global SDK state (task 2.5).

No mocks. Verifies the exit gate:
* N concurrent jobs, each in its own ``job_scope`` + ``asyncio.create_task``,
  show **no cross-job leakage** of connection / project_id / tool registry /
  normalize flags — even when all are inside their scope simultaneously.
* Outside a ``job_scope`` (the common single-agent path) every accessor uses the
  process global — behavior unchanged. (The rest of the suite staying green is
  the broader backward-compat proof.)
"""

from __future__ import annotations

import asyncio

import fastaiagent as fa
from fastaiagent._internal import project as _project
from fastaiagent._internal.scope import UNSET, scoped_normalize
from fastaiagent.client import get_connection
from fastaiagent.tool import FunctionTool
from fastaiagent.tool.registry import ToolRegistry
from fastaiagent.trace import storage as _storage


def _effective_normalize() -> bool:
    """Read the effective normalize flag the way storage.on_end does."""
    v = scoped_normalize.get()
    return _storage._normalize_enabled if v is UNSET else v


class _Barrier:
    """Tiny asyncio barrier (asyncio.Barrier is 3.11+; we support 3.10)."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._count = 0
        self._event = asyncio.Event()

    async def wait(self) -> None:
        self._count += 1
        if self._count >= self._n:
            self._event.set()
        await self._event.wait()


def test_concurrent_jobs_have_no_cross_job_leakage():
    N = 6

    async def _job(i: int, results: dict, barrier: _Barrier) -> None:
        with fa.job_scope(
            api_key=f"key{i}",
            target=f"http://h{i}",
            project=f"proj{i}",
            normalize=(i % 2 == 0),
        ):
            # A tool built inside the scope registers job-locally; every job
            # uses the SAME name to prove they don't clobber each other.
            FunctionTool(name="shared", fn=lambda i=i: f"job{i}")
            # Force every job to be inside its scope at the same time, then read.
            await barrier.wait()
            results[i] = {
                "api_key": get_connection().api_key,
                "target": get_connection().target,
                "project": _project.safe_get_project_id(),
                "tool": ToolRegistry.get("shared").fn(),
                "normalize": _effective_normalize(),
            }

    async def _run() -> dict:
        results: dict = {}
        barrier = _Barrier(N)
        tasks = [asyncio.create_task(_job(i, results, barrier)) for i in range(N)]
        await asyncio.gather(*tasks)
        return results

    # Pin a known global so we can also assert restoration afterwards.
    _project.set_project_id("GLOBAL")
    try:
        results = asyncio.run(_run())
    finally:
        _project.reset_for_testing()

    assert len(results) == N
    for i in range(N):
        assert results[i]["api_key"] == f"key{i}", results[i]
        assert results[i]["target"] == f"http://h{i}", results[i]
        assert results[i]["project"] == f"proj{i}", results[i]
        assert results[i]["tool"] == f"job{i}", results[i]
        assert results[i]["normalize"] == (i % 2 == 0), results[i]


def test_no_scope_uses_global_and_scope_resets_on_exit():
    _project.set_project_id("GLOBAL")
    FunctionTool(name="bw_compat", fn=lambda: "global")
    try:
        # No scope -> process globals (today's behavior).
        assert get_connection().api_key is None
        assert _project.safe_get_project_id() == "GLOBAL"
        assert ToolRegistry.get("bw_compat").fn() == "global"
        assert _effective_normalize() is False

        with fa.job_scope(api_key="k", project="P", normalize=True):
            FunctionTool(name="bw_compat", fn=lambda: "job")
            assert get_connection().api_key == "k"
            assert _project.safe_get_project_id() == "P"
            assert ToolRegistry.get("bw_compat").fn() == "job"
            assert _effective_normalize() is True

        # Everything restored after the scope exits.
        assert get_connection().api_key is None
        assert _project.safe_get_project_id() == "GLOBAL"
        assert ToolRegistry.get("bw_compat").fn() == "global"
        assert _effective_normalize() is False
    finally:
        _project.reset_for_testing()


def test_tool_overlay_falls_back_to_global_for_unscoped_names():
    FunctionTool(name="only_global", fn=lambda: "g")
    with fa.job_scope(tools=[FunctionTool(name="job_only", fn=lambda: "j")]):
        # Job tool resolves...
        assert ToolRegistry.get("job_only").fn() == "j"
        # ...and a name only in the global registry still resolves (overlay).
        assert ToolRegistry.get("only_global").fn() == "g"
