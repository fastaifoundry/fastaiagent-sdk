"""Example 71: Run concurrent jobs in one process with job_scope().

A runner executes many jobs in one process for one tenant. ``fa.job_scope(...)``
request-scopes the SDK's process-global state — the ``connect()`` connection,
the tool registry, the project id, and the trace-normalize flags — so concurrent
jobs don't clobber each other. Outside a ``job_scope`` (the normal single-agent
path) everything uses the process global, unchanged.

ContextVars are async-task-local, so each job must run in its own asyncio task.

Runnable as pytest (no API keys, no network):
    pytest examples/71_job_scope.py -v
"""

from __future__ import annotations

import asyncio

import fastaiagent as fa
from fastaiagent.client import get_connection
from fastaiagent.tool import FunctionTool
from fastaiagent.tool.registry import ToolRegistry


async def _job(label: str, out: dict) -> None:
    # Each job overrides the connection and seeds a job-local tool of the SAME
    # name as its siblings — they must not clobber one another.
    with fa.job_scope(api_key=f"key-{label}", project=f"proj-{label}"):
        FunctionTool(name="lookup", fn=lambda label=label: f"answer-from-{label}")
        await asyncio.sleep(0.01)  # interleave with the sibling jobs
        out[label] = {
            "key": get_connection().api_key,
            "tool": ToolRegistry.get("lookup").fn(),
        }


def test_concurrent_jobs_are_isolated() -> None:
    async def run() -> dict:
        out: dict = {}
        # One asyncio task per job — required for ContextVar isolation to hold.
        await asyncio.gather(*(asyncio.create_task(_job(x, out)) for x in "ABC"))
        return out

    out = asyncio.run(run())
    assert out["A"] == {"key": "key-A", "tool": "answer-from-A"}
    assert out["B"] == {"key": "key-B", "tool": "answer-from-B"}
    assert out["C"] == {"key": "key-C", "tool": "answer-from-C"}


def test_outside_a_scope_is_unchanged() -> None:
    # No job_scope -> the process-global (here: disconnected) connection.
    assert get_connection().api_key is None
