"""HTTP-level tests for the v1.14.1 ``tool_overrides`` shape on
``PATCH /api/replay/forks/{fork_id}``.

The pre-v1.14.1 ``tool_response`` field was silently dropped for agent
reruns — the UI looked like it worked but the override never reached
the rerun. v1.14.1 introduced ``tool_overrides: {name: response}`` that
the route wires through ``ForkedReplay.with_tool_override`` with a
stub :class:`FunctionTool`.

These tests use the real FastAPI app + real SQLite (matches the
project's no-mocking rule) and exercise the route directly to confirm
the wiring lands on the fork's internal state.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.routes.replay import _clear_forks_for_tests, _get_fork  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


def _seed_trace(db_path: Path, trace_id: str = "trace-tool-override") -> None:
    db = init_local_db(db_path)
    now = datetime.now(tz=timezone.utc).isoformat()
    attrs = {
        "agent.name": "demo",
        "agent.input": "Look up ORD-1",
        "agent.output": "Order ORD-1 was delivered.",
        "agent.system_prompt": "be helpful",
        "agent.config": json.dumps({"max_iterations": 5}),
        "agent.tools": json.dumps([{"name": "lookup_order", "tool_type": "function"}]),
        "agent.guardrails": json.dumps([]),
        "agent.llm.provider": "openai",
        "agent.llm.model": "gpt-4o-mini",
        "agent.llm.config": json.dumps(
            {"provider": "openai", "model": "gpt-4o-mini", "api_key": "k"}
        ),
    }
    try:
        db.execute(
            """INSERT INTO spans
               (span_id, trace_id, parent_span_id, name, start_time, end_time,
                status, attributes, events)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "root",
                trace_id,
                None,
                "agent.demo",
                now,
                now,
                "OK",
                json.dumps(attrs),
                "[]",
            ),
        )
    finally:
        db.close()


@pytest.fixture
def client(tmp_path: Path):
    db_path = tmp_path / "local.db"
    _seed_trace(db_path)
    _clear_forks_for_tests()
    app = build_app(db_path=str(db_path), no_auth=True)
    yield TestClient(app)
    _clear_forks_for_tests()


def _fork(client: TestClient, trace_id: str = "trace-tool-override") -> str:
    r = client.post(f"/api/replay/{trace_id}/fork", json={"step": 0})
    assert r.status_code == 200, r.text
    return r.json()["fork_id"]


class TestToolOverridesWiring:
    def test_route_records_override_on_the_fork(self, client: TestClient) -> None:
        # v1.14.1: tool_overrides should install a stub FunctionTool on
        # the fork via with_tool_override. Confirm by inspecting the
        # fork's _tool_overrides dict directly (no LLM call needed).
        fork_id = _fork(client)
        r = client.patch(
            f"/api/replay/forks/{fork_id}",
            json={"tool_overrides": {"lookup_order": {"result": "stubbed value"}}},
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "modified"

        fork = _get_fork(fork_id)
        assert "lookup_order" in fork._tool_overrides
        stub_tool = fork._tool_overrides["lookup_order"]
        assert stub_tool.name == "lookup_order"
        # The stub's fn returns the canned response regardless of args.
        result = stub_tool.fn(any_arg="ignored", another="ignored")
        assert result == {"result": "stubbed value"}

    def test_multiple_overrides_are_all_installed(self, client: TestClient) -> None:
        fork_id = _fork(client)
        r = client.patch(
            f"/api/replay/forks/{fork_id}",
            json={
                "tool_overrides": {
                    "lookup_order": {"result": "stub1"},
                    "create_ticket": {"id": "T-9"},
                }
            },
        )
        assert r.status_code == 200, r.text
        fork = _get_fork(fork_id)
        assert set(fork._tool_overrides.keys()) == {"lookup_order", "create_ticket"}

    def test_deprecated_tool_response_surfaces_warning_and_does_nothing(
        self, client: TestClient
    ) -> None:
        # Pre-v1.14.1 the UI sent this and the agent rerun silently
        # ignored it. v1.14.1 keeps accepting the field for forward-compat
        # but returns a deprecation warning and installs no override.
        fork_id = _fork(client)
        r = client.patch(
            f"/api/replay/forks/{fork_id}",
            json={"tool_response": {"would-be": "override"}},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "modified"
        assert "deprecation_warnings" in body
        assert any("tool_overrides" in w for w in body["deprecation_warnings"])

        # No actual override installed — caller must migrate to the new field.
        fork = _get_fork(fork_id)
        assert fork._tool_overrides == {}

    def test_other_modifications_still_work_alongside_overrides(self, client: TestClient) -> None:
        # Sanity that the new branch didn't break prompt/input/config.
        fork_id = _fork(client)
        r = client.patch(
            f"/api/replay/forks/{fork_id}",
            json={
                "prompt": "Be concise.",
                "tool_overrides": {"lookup_order": {"x": 1}},
                "config": {"temperature": 0.0},
            },
        )
        assert r.status_code == 200, r.text
        fork = _get_fork(fork_id)
        assert fork._modifications.get("prompt") == "Be concise."
        assert "lookup_order" in fork._tool_overrides
        assert fork._modifications.get("config") == {"temperature": 0.0}
