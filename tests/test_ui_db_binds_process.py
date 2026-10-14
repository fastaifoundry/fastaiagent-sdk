"""``fastaiagent ui --db X`` makes X the store for the whole process (1.89.0, P4).

``build_app(db_path=X)`` scoped the UI's routes, but the tracer, the
checkpointers and every other SDK store resolved their own path from config.
Under ``--db X`` a Playground run's trace went to the configured store (its link
404'd), a replay rerun landed where compare could not see it, and resuming an
approval looked for its checkpoint in the wrong file and answered 409.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    for name in (
        "FASTAIAGENT_LOCAL_DB",
        "FASTAIAGENT_TRACE_DB_PATH",
        "FASTAIAGENT_CHECKPOINT_DB_PATH",
        "FASTAIAGENT_API_KEY",
        "FASTAIAGENT_TARGET",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    otel.reset()
    yield monkeypatch
    otel.reset()
    reset_config()


def test_db_flag_routes_traces_and_checkpoints_to_x(clean_env: Any, tmp_path: Path) -> None:
    import fastaiagent as fa
    from fastaiagent._internal.config import get_config
    from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
    from fastaiagent.cli.ui import _bind_process_to_db
    from fastaiagent.testing import TestModel

    elsewhere = tmp_path / "elsewhere.db"
    clean_env.setenv("FASTAIAGENT_LOCAL_DB", str(elsewhere))
    clean_env.setenv("FASTAIAGENT_TRACE_DB_PATH", str(elsewhere))
    x = tmp_path / "x" / "local.db"
    x.parent.mkdir()

    _bind_process_to_db(x)

    cfg = get_config()
    assert Path(cfg.local_db_path).resolve() == x.resolve()
    assert Path(cfg.resolved_trace_db_path).resolve() == x.resolve()
    assert Path(SQLiteCheckpointer().db_path).resolve() == x.resolve()

    result = fa.Agent(name="bound", system_prompt="s", llm=TestModel(response="ok")).run("hi")
    with sqlite3.connect(x) as conn:
        found = conn.execute(
            "SELECT COUNT(*) FROM spans WHERE trace_id = ?", (result.trace_id,)
        ).fetchone()[0]
    assert found, "the trace was not written to the --db store"


def test_an_embedder_with_split_stores_is_told_where_the_trace_went(
    clean_env: Any, tmp_path: Path, capture_server: Any
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.server import build_app

    clean_env.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "tracer.db"))
    clean_env.setenv("FASTAIAGENT_ALLOW_PRIVATE_NETWORKS", "1")
    capture_server.set_response_sequence(
        [
            (
                200,
                {
                    "id": "c1",
                    "object": "chat.completion",
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hi"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        ]
    )
    client = TestClient(build_app(db_path=str(tmp_path / "ui.db"), no_auth=True))
    r = client.post(
        "/api/playground/run",
        json={
            "provider": "custom",
            "model": "m",
            "prompt_template": "hi",
            "variables": {},
            "base_url": capture_server.url,
            "api_key": "stub",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["trace_in_ui_db"] is False
