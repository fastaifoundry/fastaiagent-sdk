"""UI route tests for /api/learned_memory — TestClient, no LLM, no mocking."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.learn import Fact, MemoryStore  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    from fastaiagent._internal.config import reset_config

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    reset_config()

    # Seed via the public API rather than raw SQL.
    store = MemoryStore(db_path=str(db_path))
    store.add(Fact(scope="agent", scope_id="alpha", fact="first agent fact"))
    store.add(Fact(scope="agent", scope_id="alpha", fact="second agent fact"))
    store.add(Fact(scope="user", scope_id="user-42", fact="prefers terse answers"))

    # Add a superseded chain.
    old_id = store.add(Fact(scope="agent", scope_id="alpha", fact="old fact"))
    new_id = store.add(Fact(scope="agent", scope_id="alpha", fact="replacement fact"))
    store.supersede(old_id, new_id)

    app = build_app(db_path=str(db_path), no_auth=True)
    return TestClient(app)


def test_list_returns_only_active_by_default(client: TestClient) -> None:
    r = client.get("/api/learned_memory")
    assert r.status_code == 200
    body = r.json()
    assert "rows" in body
    facts = {row["fact"] for row in body["rows"]}
    assert "first agent fact" in facts
    assert "second agent fact" in facts
    assert "replacement fact" in facts
    assert "old fact" not in facts  # superseded — excluded by default


def test_list_with_include_superseded(client: TestClient) -> None:
    r = client.get("/api/learned_memory?include_superseded=true")
    assert r.status_code == 200
    facts = {row["fact"] for row in r.json()["rows"]}
    assert "old fact" in facts
    assert "replacement fact" in facts


def test_filter_by_scope(client: TestClient) -> None:
    r = client.get("/api/learned_memory?scope=user")
    assert r.status_code == 200
    body = r.json()
    facts = {row["fact"] for row in body["rows"]}
    assert facts == {"prefers terse answers"}


def test_filter_by_scope_and_id(client: TestClient) -> None:
    r = client.get("/api/learned_memory?scope=agent&scope_id=alpha")
    assert r.status_code == 200
    facts = {row["fact"] for row in r.json()["rows"]}
    # active agent/alpha facts only
    assert "first agent fact" in facts
    assert "replacement fact" in facts
    assert "prefers terse answers" not in facts


def test_invalid_scope_returns_empty_with_error(client: TestClient) -> None:
    r = client.get("/api/learned_memory?scope=bogus")
    assert r.status_code == 200
    body = r.json()
    assert body["rows"] == []
    assert "error" in body


def test_scopes_endpoint_groups_distinct_pairs(client: TestClient) -> None:
    r = client.get("/api/learned_memory/scopes")
    assert r.status_code == 200
    pairs = r.json()["scopes"]
    pair_set = {(p["scope"], p["scope_id"]) for p in pairs}
    assert ("agent", "alpha") in pair_set
    assert ("user", "user-42") in pair_set


def test_source_trace_id_round_trips(tmp_path: Path, monkeypatch) -> None:
    """A fact stamped with a source_trace_id surfaces it for the UI Source column."""
    from fastaiagent._internal.config import reset_config

    db_path = tmp_path / "src.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    reset_config()
    store = MemoryStore(db_path=str(db_path))
    store.add(
        Fact(
            scope="user",
            scope_id="u",
            fact="auto-extracted fact",
            source_trace_id="abc123def456",
            confidence=0.6,
        )
    )
    tc = TestClient(build_app(db_path=str(db_path), no_auth=True))
    row = next(
        r
        for r in tc.get("/api/learned_memory").json()["rows"]
        if r["fact"] == "auto-extracted fact"
    )
    assert row["source_trace_id"] == "abc123def456"
    assert row["confidence"] == 0.6


def test_superseded_chain_exposed(client: TestClient) -> None:
    """include_superseded reveals the replaced row with superseded_by populated."""
    rows = client.get("/api/learned_memory?include_superseded=true").json()["rows"]
    superseded = [r for r in rows if r["fact"] == "old fact"]
    assert superseded, "old fact not returned with include_superseded"
    assert superseded[0]["superseded_by"] is not None


def test_redact_param_is_noop_without_policy(client: TestClient) -> None:
    # ?redact=true is honored only when a read-mode RedactionPolicy is installed.
    from fastaiagent.trace.redaction import set_redaction_policy

    set_redaction_policy(None)
    r = client.get("/api/learned_memory?redact=true")
    assert r.status_code == 200
    facts = {row["fact"] for row in r.json()["rows"]}
    assert "first agent fact" in facts  # unchanged — no policy installed


def test_redact_param_masks_facts_with_read_policy(client: TestClient) -> None:
    from fastaiagent.trace.redaction import RedactionPolicy, set_redaction_policy

    set_redaction_policy(
        RedactionPolicy(patterns=[r"terse"], replacement="[REDACTED]", mode="read")
    )
    try:
        r = client.get("/api/learned_memory?redact=true&scope=user")
        assert r.status_code == 200
        facts = {row["fact"] for row in r.json()["rows"]}
        assert "prefers [REDACTED] answers" in facts
        assert "prefers terse answers" not in facts
    finally:
        set_redaction_policy(None)
