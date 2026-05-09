"""Tests for the ``/api/providers`` UI route.

Exercises the FastAPI route directly via TestClient — no browser, no
network. Validates the response shape and that built-ins + presets both
appear.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# fastapi ships in the [ui] extra; skip cleanly when running against
# the base install (matches other UI-dependent test modules).
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.ui.server import build_app  # noqa: E402


@pytest.fixture
def app(tmp_path: Path):  # type: ignore[no-untyped-def]
    db_path = tmp_path / "local.db"
    return build_app(db_path=str(db_path), no_auth=True)


def test_providers_endpoint_returns_builtins_and_presets(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        resp = client.get("/api/providers")
    assert resp.status_code == 200
    payload = resp.json()
    assert "providers" in payload
    assert "reserved" in payload

    keys = [p["key"] for p in payload["providers"]]
    # Built-ins
    for must in ("openai", "anthropic", "ollama", "azure", "bedrock", "custom", "test"):
        assert must in keys
    # Seed presets
    for must in ("groq", "gemini", "openrouter", "deepseek", "mistral"):
        assert must in keys


def test_providers_entry_shape(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        resp = client.get("/api/providers")
    payload = resp.json()
    groq = next(p for p in payload["providers"] if p["key"] == "groq")
    assert groq["base_url"] == "https://api.groq.com/openai/v1"
    assert groq["env_var"] == "GROQ_API_KEY"
    assert groq["wire"] == "openai_compat"
    assert groq["builtin"] is False
    assert isinstance(groq["capabilities"], dict)

    openai = next(p for p in payload["providers"] if p["key"] == "openai")
    assert openai["builtin"] is True


def test_providers_reserved_set_includes_test(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        resp = client.get("/api/providers")
    payload = resp.json()
    assert "test" in payload["reserved"]
    # The reserved list and the providers list must be consistent: every
    # reserved key shows up in providers as a builtin.
    builtin_keys = {p["key"] for p in payload["providers"] if p["builtin"]}
    for key in payload["reserved"]:
        assert key in builtin_keys
