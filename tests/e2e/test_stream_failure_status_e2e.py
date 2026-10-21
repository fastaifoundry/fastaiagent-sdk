"""Live: a failed streamed call is traced exactly like a failed non-streamed one (P5).

A real OpenAI 404 (an unknown model) through ``acomplete`` and ``astream``, an
agent's ``stream`` / ``astream``, and the Playground's ``/stream`` route; plus a
real successful stream as the control. Spans are read straight from the
``local.db`` the process writes (storage is synchronous: no sleeps).
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

BAD_MODEL = "gpt-5-mini-ws1-nonexistent"


@pytest.fixture
def traced_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    monkeypatch.delenv("FASTAIAGENT_API_KEY", raising=False)
    monkeypatch.delenv("FASTAIAGENT_TARGET", raising=False)
    reset_config()
    _project.set_project_id("e2e-p5")
    otel.reset()
    try:
        yield db_path
    finally:
        otel.reset()
        _project.reset_for_testing()
        reset_config()


def _llm_spans(db: Any) -> list[tuple[str, str, list[str]]]:
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT name, status, events FROM spans WHERE name LIKE 'llm.%' ORDER BY start_time"
        ).fetchall()
    return [
        (
            name,
            status,
            [
                e["attributes"].get("exception.type")
                for e in json.loads(ev or "[]")
                if e.get("name") == "exception"
            ],
        )
        for name, status, ev in rows
    ]


async def _drain(gen: Any) -> None:
    async for _ in gen:
        pass


def test_complete_and_stream_fail_the_same_way(traced_db: Any) -> None:
    from fastaiagent.llm import LLMClient, UserMessage

    client = LLMClient(model=BAD_MODEL)
    with pytest.raises(Exception):  # noqa: B017
        asyncio.run(client.acomplete([UserMessage("hi")]))
    with pytest.raises(Exception):  # noqa: B017
        asyncio.run(_drain(client.astream([UserMessage("hi")])))

    spans = _llm_spans(traced_db)
    assert len(spans) == 2, spans
    (_, s1, e1), (_, s2, e2) = spans
    assert s1 == s2 == "ERROR", spans
    assert e1 and e2 and e1[-1] == e2[-1], spans


@pytest.mark.parametrize("how", ["stream", "astream"])
def test_an_agent_stream_failure_is_an_error_trace(traced_db: Any, how: str) -> None:
    import fastaiagent as fa
    from fastaiagent.llm import LLMClient

    agent = fa.Agent(name="p5-live", system_prompt="s", llm=LLMClient(model=BAD_MODEL))
    with pytest.raises(Exception):  # noqa: B017
        if how == "stream":
            agent.stream("hi")
        else:
            asyncio.run(_drain(agent.astream("hi")))
    with sqlite3.connect(traced_db) as conn:
        statuses = dict(conn.execute("SELECT name, status FROM spans").fetchall())
    assert statuses.get("agent.p5-live") == "ERROR", statuses
    assert all(s == "ERROR" for n, s in statuses.items() if n.startswith("llm.")), statuses


def test_the_playground_stream_failure_is_an_error_trace(traced_db: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.db import init_local_db
    from fastaiagent.ui.server import build_app

    init_local_db(traced_db).close()
    client = TestClient(build_app(db_path=str(traced_db), no_auth=True))
    r = client.post(
        "/api/playground/stream",
        json={"provider": "openai", "model": BAD_MODEL, "prompt_template": "hi", "variables": {}},
    )
    assert "event: error" in r.text, r.text
    with sqlite3.connect(traced_db) as conn:
        root = conn.execute("SELECT status FROM spans WHERE name = 'playground.run'").fetchone()
    assert root and root[0] == "ERROR", root


def test_a_real_successful_stream_is_not_an_error(traced_db: Any) -> None:
    from fastaiagent.llm import LLMClient, UserMessage

    llm = LLMClient(model="gpt-5-mini", reasoning="minimal")
    asyncio.run(_drain(llm.astream([UserMessage("Say: ok")])))
    spans = _llm_spans(traced_db)
    assert spans and all(s != "ERROR" for _, s, _ in spans), spans
