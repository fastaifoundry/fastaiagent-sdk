"""Tests for the configurable Playground model catalog (1.53.0).

Model ids rot on the provider's schedule, not ours. These cover the override
file that lets a user fix a stale dropdown without an SDK release, and the
guarantee that a broken file degrades to the shipped defaults rather than
taking the model picker offline.

Real files, real FastAPI app. No mocking.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.ui import model_catalog  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


@pytest.fixture
def db_path(temp_dir: Path) -> Path:
    path = temp_dir / ".fastaiagent" / "local.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    init_local_db(path).close()
    return path


@pytest.fixture
def client(db_path: Path) -> TestClient:
    return TestClient(build_app(db_path=str(db_path), no_auth=True))


def write_catalog(db_path: Path, payload: object) -> Path:
    target = db_path.parent / model_catalog.CATALOG_FILENAME
    target.write_text(
        payload if isinstance(payload, str) else json.dumps(payload),
        encoding="utf-8",
    )
    return target


def models_for(client: TestClient, provider: str) -> list[str]:
    rows = client.get("/api/playground/models").json()["providers"]
    return next(r for r in rows if r["provider"] == provider)["models"]


class TestCatalogPath:
    def test_resolves_next_to_local_db(self, db_path: Path) -> None:
        assert model_catalog.catalog_path(str(db_path)) == (
            db_path.parent / "models.json"
        )

    def test_env_var_wins(
        self, db_path: Path, monkeypatch: pytest.MonkeyPatch, temp_dir: Path
    ) -> None:
        explicit = temp_dir / "elsewhere.json"
        monkeypatch.setenv(model_catalog.CATALOG_ENV_VAR, str(explicit))
        assert model_catalog.catalog_path(str(db_path)) == explicit


class TestDefaults:
    def test_no_file_yields_shipped_defaults(self, client: TestClient) -> None:
        assert models_for(client, "anthropic") == (
            model_catalog.BUILTIN_MODELS["anthropic"]
        )

    def test_retired_models_are_not_shipped(self, client: TestClient) -> None:
        """These 404 against the live APIs; they must not be offered."""
        retired = {
            "anthropic": "claude-3-5-sonnet-latest",
            "gemini": "gemini-2.5-pro",
        }
        for provider, dead_model in retired.items():
            assert dead_model not in models_for(client, provider)

    def test_decommissioned_groq_models_are_not_shipped(
        self, client: TestClient
    ) -> None:
        groq = models_for(client, "groq")
        for dead in (
            "llama-3.1-70b-versatile",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ):
            assert dead not in groq, f"{dead} was decommissioned by Groq"

    def test_preset_default_model_leads_the_list(self, client: TestClient) -> None:
        from fastaiagent.llm.providers import get_preset

        preset = get_preset("groq")
        assert preset is not None
        assert models_for(client, "groq")[0] == preset.default_model


class TestOverrides:
    def test_object_form_replaces_the_list(
        self, client: TestClient, db_path: Path
    ) -> None:
        write_catalog(db_path, {"anthropic": {"models": ["claude-opus-5"]}})
        assert models_for(client, "anthropic") == ["claude-opus-5"]

    def test_bare_list_shorthand(self, client: TestClient, db_path: Path) -> None:
        write_catalog(db_path, {"openai": ["gpt-5.2", "gpt-4o-mini"]})
        assert models_for(client, "openai") == ["gpt-5.2", "gpt-4o-mini"]

    def test_untouched_providers_keep_defaults(
        self, client: TestClient, db_path: Path
    ) -> None:
        write_catalog(db_path, {"openai": ["gpt-4o-mini"]})
        assert models_for(client, "anthropic") == (
            model_catalog.BUILTIN_MODELS["anthropic"]
        )

    def test_preset_provider_can_be_overridden(
        self, client: TestClient, db_path: Path
    ) -> None:
        write_catalog(db_path, {"groq": ["openai/gpt-oss-20b"]})
        assert models_for(client, "groq") == ["openai/gpt-oss-20b"]

    def test_duplicates_and_blanks_are_cleaned(
        self, client: TestClient, db_path: Path
    ) -> None:
        write_catalog(db_path, {"openai": ["gpt-4o-mini", "  ", "gpt-4o-mini", "gpt-4o"]})
        assert models_for(client, "openai") == ["gpt-4o-mini", "gpt-4o"]

    def test_env_var_override_changes_has_key(
        self, client: TestClient, db_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MY_CUSTOM_KEY", raising=False)
        write_catalog(db_path, {"openai": {"models": ["x"], "env_var": "MY_CUSTOM_KEY"}})
        rows = client.get("/api/playground/models").json()["providers"]
        openai_row = next(r for r in rows if r["provider"] == "openai")
        assert openai_row["env_var"] == "MY_CUSTOM_KEY"
        assert openai_row["has_key"] is False

        monkeypatch.setenv("MY_CUSTOM_KEY", "sk-test")
        rows = client.get("/api/playground/models").json()["providers"]
        assert next(r for r in rows if r["provider"] == "openai")["has_key"] is True


class TestMalformedFileDegradesGracefully:
    """A typo in a JSON file must never take the model picker offline."""

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param("{ not json at all", id="invalid-json"),
            pytest.param(json.dumps(["a", "list"]), id="top-level-list"),
            pytest.param(json.dumps({"anthropic": 42}), id="entry-not-list-or-object"),
            pytest.param(
                json.dumps({"anthropic": {"models": "not-a-list"}}), id="models-not-list"
            ),
            pytest.param(
                json.dumps({"anthropic": {"models": [1, 2, 3]}}), id="models-not-strings"
            ),
        ],
    )
    def test_falls_back_to_builtins(
        self, client: TestClient, db_path: Path, payload: str
    ) -> None:
        write_catalog(db_path, payload)
        assert models_for(client, "anthropic") == (
            model_catalog.BUILTIN_MODELS["anthropic"]
        )

    def test_unknown_provider_is_skipped_others_survive(
        self, client: TestClient, db_path: Path
    ) -> None:
        write_catalog(
            db_path,
            {"notaprovider": ["x"], "anthropic": ["claude-opus-5"]},
        )
        rows = client.get("/api/playground/models").json()["providers"]
        assert "notaprovider" not in {r["provider"] for r in rows}
        assert models_for(client, "anthropic") == ["claude-opus-5"]

    def test_pricing_key_is_not_treated_as_a_provider(
        self, client: TestClient, db_path: Path
    ) -> None:
        write_catalog(
            db_path,
            {"pricing": {"gpt-4o-mini": {"input_per_1m": 1, "output_per_1m": 2}}},
        )
        rows = client.get("/api/playground/models").json()["providers"]
        assert "pricing" not in {r["provider"] for r in rows}
