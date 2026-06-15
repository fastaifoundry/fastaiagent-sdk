"""E2E (Task A) — run a real live_playground command against the LIVE plane.

Unlike ``test_runner_trace_push.py`` (which drains to a localhost stand-in), this
hits a real Enterprise plane: it requires ``FASTAIAGENT_API_KEY`` +
``FASTAIAGENT_TARGET`` (and ``OPENAI_API_KEY``), so it SKIPS cleanly when those
aren't provided (the standard PLATFORM_ENV gate). Provision an SDK API key on the
plane and point ``FASTAIAGENT_TARGET`` at it (e.g. ``http://localhost:20001``).

It connects exactly as the shipped ``fastaiagent runner`` CLI now does, runs the
runner's ``execute_command`` with a real ``gpt-4o-mini`` agent, flushes the real
exporter the way the daemon does, and asserts the plane accepted the push
(``count_unsynced == 0`` after a confirmed 2xx) and the runner reported a trace_id.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


@pytest.fixture
def _clean_platform():
    import fastaiagent
    from fastaiagent.client import _connection
    from fastaiagent.trace import otel

    yield
    try:
        fastaiagent.disconnect()
    except Exception:
        pass
    otel.reset()
    _connection.api_key = None
    _connection.target = "https://app.fastaiagent.net"
    _connection.project = None
    _connection.project_id = None
    _connection.domain_id = None
    _connection._platform_processor = None


def test_runner_live_playground_pushes_to_live_plane(isolated_local_db, _clean_platform) -> None:
    require_env()  # OPENAI_API_KEY + FASTAIAGENT_API_KEY + FASTAIAGENT_TARGET (skips if absent)

    import fastaiagent
    from fastaiagent import Agent, LLMClient
    from fastaiagent.client import _connection
    from fastaiagent.runner.execute import execute_command
    from fastaiagent.trace import otel

    otel.reset()
    fastaiagent.connect(
        api_key=os.environ["FASTAIAGENT_API_KEY"], target=os.environ["FASTAIAGENT_TARGET"]
    )
    assert _connection.is_connected

    agent = Agent(
        name="taskA-harness",
        system_prompt="Reply with exactly the word OK and nothing else.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    )
    cmd = {
        "command_id": "taskA-harness-1",
        "type": "live_playground",
        "tenant": _connection.domain_id,
        "payload": {"agent": agent.to_dict(), "input": "say ok"},
    }

    async def _drive():
        # Own asyncio task (the daemon's 2.6 invariant), then flush the exporter
        # the way the daemon's _run_command does.
        res = await asyncio.create_task(execute_command(cmd))
        proc = _connection._platform_processor
        if proc is not None:
            await asyncio.to_thread(proc.force_flush, 10000)
        return res

    res = asyncio.run(_drive())

    assert res.status == "completed", res
    assert res.trace_id, f"runner must report a trace_id: {res}"

    # The real plane accepted the push (drain marks synced only on a 2xx).
    from fastaiagent.trace.storage import TraceStore

    store = TraceStore()
    assert store.count_unsynced("test-proj") == 0
    store.close()
