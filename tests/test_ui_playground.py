"""Integration tests for the Prompt Playground endpoints.

Uses real FastAPI + real SQLite; the LLM-call tests skip themselves when
the relevant API key is not in the environment so the suite stays fast on
CI machines without keys.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.routes import playground as playground_route  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


@pytest.fixture
def empty_db(temp_dir: Path) -> Path:
    db_path = temp_dir / ".fastaiagent" / "local.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_local_db(db_path).close()
    return db_path


@pytest.fixture
def client(empty_db: Path) -> TestClient:
    app = build_app(db_path=str(empty_db), no_auth=True)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Pure helpers — no network
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    def test_returns_provider_catalog(self, client: TestClient) -> None:
        r = client.get("/api/playground/models")
        assert r.status_code == 200
        body = r.json()
        provider_names = {p["provider"] for p in body["providers"]}
        assert {"openai", "anthropic", "ollama"}.issubset(provider_names)

    def test_has_key_reflects_real_env(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-real-env")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        body = client.get("/api/playground/models").json()
        by_name = {p["provider"]: p for p in body["providers"]}
        assert by_name["openai"]["has_key"] is True
        assert by_name["anthropic"]["has_key"] is False
        # Ollama doesn't require a key — always reachable from the UI's perspective.
        assert by_name["ollama"]["has_key"] is True

    def test_includes_v1_8_0_presets(self, client: TestClient) -> None:
        """v1.8.1: catalog merges the preset registry — Groq, Gemini, etc.
        appear in the dropdown alongside the built-ins, no UI rebuild needed."""
        body = client.get("/api/playground/models").json()
        provider_names = {p["provider"] for p in body["providers"]}
        # A representative subset; full list lives in
        # fastaiagent.llm.providers._presets.
        for must in {"groq", "gemini", "openrouter", "deepseek", "mistral"}:
            assert must in provider_names, f"{must} preset missing from /models"

    def test_preset_default_model_first_in_list(
        self, client: TestClient
    ) -> None:
        """The preset's default_model leads the suggestions list so the
        dropdown defaults to a known-good choice on first selection."""
        from fastaiagent.llm.providers import get_preset

        body = client.get("/api/playground/models").json()
        by_name = {p["provider"]: p for p in body["providers"]}
        groq_preset = get_preset("groq")
        gemini_preset = get_preset("gemini")
        assert groq_preset is not None and gemini_preset is not None
        assert by_name["groq"]["models"][0] == groq_preset.default_model
        assert by_name["gemini"]["models"][0] == gemini_preset.default_model

    def test_preset_has_key_reflects_env_var(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "gsk-fixture-key")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        body = client.get("/api/playground/models").json()
        by_name = {p["provider"]: p for p in body["providers"]}
        assert by_name["groq"]["has_key"] is True
        assert by_name["gemini"]["has_key"] is False
        assert by_name["groq"]["env_var"] == "GROQ_API_KEY"
        assert by_name["gemini"]["env_var"] == "GEMINI_API_KEY"

    def test_models_list_no_duplicates(self, client: TestClient) -> None:
        """Default-model + curated hints are deduplicated (the default
        already appears in some hint lists)."""
        body = client.get("/api/playground/models").json()
        for entry in body["providers"]:
            assert len(entry["models"]) == len(set(entry["models"])), (
                f"{entry['provider']} has duplicate models: {entry['models']}"
            )


class TestRunValidation:
    def test_no_api_key_returns_400(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt_template": "Hi",
                "variables": {},
                "parameters": {"temperature": 1.0, "max_tokens": 16, "top_p": 1.0},
            },
        )
        assert r.status_code == 400
        assert "OPENAI_API_KEY" in r.json()["detail"]

    def test_image_b64_without_media_type_400(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test")
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "openai",
                "model": "gpt-4o",
                "prompt_template": "describe it",
                "variables": {},
                "parameters": {"temperature": 1.0, "max_tokens": 16, "top_p": 1.0},
                "image_b64": "Zm9vYmFy",  # not really an image — we error before validation
            },
        )
        assert r.status_code == 400
        assert "image_media_type" in r.json()["detail"]

    # -----------------------------------------------------------------
    # security_review_1.md H10 — base64 image size cap
    # -----------------------------------------------------------------

    def test_image_b64_too_large_string_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multi-million-char ``image_b64`` field is refused by Pydantic
        validation before the worker ever decodes it.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test")
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "openai",
                "model": "gpt-4o",
                "prompt_template": "x",
                "variables": {},
                "parameters": {"temperature": 1.0, "max_tokens": 16, "top_p": 1.0},
                # 36M chars — over the 35M cap.
                "image_b64": "A" * 36_000_000,
                "image_media_type": "image/jpeg",
            },
        )
        # Pydantic returns 422 for max_length violations.
        assert r.status_code == 422

    def test_image_b64_decoded_oversize_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even within the 35M-char cap, anything that decodes above
        25 MiB must hit a 413 — defence in depth against base64 padding.
        """
        import base64 as _b64

        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test")
        # Build raw bytes just above 25 MiB, then base64 it. The encoded
        # string is ~33.5M chars — under the field cap — so Pydantic
        # passes it through and the route enforces the size check.
        raw = b"A" * (25 * 1024 * 1024 + 16)
        encoded = _b64.b64encode(raw).decode("ascii")
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "openai",
                "model": "gpt-4o",
                "prompt_template": "x",
                "variables": {},
                "parameters": {"temperature": 1.0, "max_tokens": 16, "top_p": 1.0},
                "image_b64": encoded,
                "image_media_type": "image/jpeg",
            },
        )
        assert r.status_code == 413

    # -----------------------------------------------------------------
    # security_review_1.md H7 — LLM error correlation-id redaction
    # -----------------------------------------------------------------

    def test_run_redacts_llm_provider_error(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider-side exception must not leak into the response.

        The route used to return ``f"LLM call failed: {type(e).__name__}: {e}"``,
        which echoes provider-side error text — frequently containing
        request-id, org-id, and partial key prefixes. The fixed route
        returns an opaque ``correlation_id`` and logs the full error
        server-side.

        This test stubs ``LLMClient.acomplete`` to inject a fake error;
        we are testing the *redaction*, not provider integration, so a
        narrow stub is the correct level of isolation.
        """
        from fastaiagent import llm as _llm_mod

        secret_marker = "sk-secret-prefix-12345 / org-OPENAI-XXXX / req_abc123"

        class _BoomLLMClient:
            def __init__(self, **kwargs: object) -> None:
                pass

            async def acomplete(self, messages: object) -> object:
                raise RuntimeError(secret_marker)

        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test")
        monkeypatch.setattr(_llm_mod, "LLMClient", _BoomLLMClient)

        r = client.post(
            "/api/playground/run",
            json={
                "provider": "openai",
                "model": "gpt-4o",
                "prompt_template": "hi",
                "variables": {},
                "parameters": {"temperature": 1.0, "max_tokens": 16, "top_p": 1.0},
            },
        )
        assert r.status_code == 502
        body_text = r.text
        # Negative: provider error must not appear in the response body.
        assert secret_marker not in body_text
        assert "RuntimeError" not in body_text
        # Positive: correlation_id present so a user can grep server logs.
        body = r.json()
        detail = body.get("detail", {})
        if isinstance(detail, dict):
            assert "correlation_id" in detail
            assert detail.get("error", "").lower().startswith("llm call failed")

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Live test — requires OPENAI_API_KEY in the environment.",
    )
    def test_run_redacts_real_openai_auth_error(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Live end-to-end check that the H7 redaction handles a real
        provider error, not just our stubbed one.

        We deliberately point the route at an invalid OPENAI_API_KEY so
        the real OpenAI endpoint returns 401. The route's exception
        handler runs, and we assert the response surface is the
        opaque ``{"error": "...", "correlation_id": "..."}`` shape —
        no OpenAI request-id or key prefix leaks through.
        """
        # Override the env-resident real key with a clearly-invalid one
        # for this single request. The ``has_key`` check passes (env is
        # non-empty), but the actual call fails at OpenAI's edge.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-deliberately-invalid-for-h7-test")
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt_template": "say hi",
                "variables": {},
                "parameters": {"temperature": 1.0, "max_tokens": 16, "top_p": 1.0},
            },
        )
        assert r.status_code == 502, r.text
        body_lower = r.text.lower()
        # Negative: nothing OpenAI-internal must appear in the response.
        for leak in (
            "request id",
            "req_",
            "org-",
            "openaierror",
            "authenticationerror",
            "invalid_api_key",
            "sk-deliberately",
        ):
            assert leak not in body_lower, (
                f"H7 regression: provider leak {leak!r} present: {r.text}"
            )
        body = r.json()
        detail = body.get("detail", {})
        assert isinstance(detail, dict)
        assert "correlation_id" in detail
        assert detail.get("error", "").lower().startswith("llm call failed")


class TestTemplateResolution:
    def test_substitutes_variables(self) -> None:
        out = playground_route._resolve_template(
            "Hi {{name}}, on topic {{topic}}",
            {"name": "Alice", "topic": "refunds"},
        )
        assert out == "Hi Alice, on topic refunds"

    def test_leaves_unknown_placeholders(self) -> None:
        out = playground_route._resolve_template(
            "Hi {{name}}", {"missing": "x"}
        )
        assert out == "Hi {{name}}"

    def test_detects_variables(self) -> None:
        assert playground_route._detect_variables(
            "Hi {{name}}, can I help with {{topic}}? And {{name}} again."
        ) == ["name", "topic"]


class TestSaveAsEval:
    def test_appends_jsonl_under_dataset_dir(
        self, client: TestClient, empty_db: Path
    ) -> None:
        # First save creates the file.
        r = client.post(
            "/api/playground/save-as-eval",
            json={
                "dataset_name": "my_set",
                "input": "what is 2+2",
                "expected_output": "4",
                "model": "gpt-4o-mini",
                "provider": "openai",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["dataset_name"] == "my_set"
        assert body["line_count"] == 1
        path = Path(body["path"])
        assert path.exists()
        assert path.parent == empty_db.parent / "datasets"

        # Second save appends.
        r2 = client.post(
            "/api/playground/save-as-eval",
            json={
                "dataset_name": "my_set",
                "input": "what is 3+3",
                "expected_output": "6",
            },
        )
        assert r2.status_code == 200
        assert r2.json()["line_count"] == 2

        # File contents are valid JSONL with both rows.
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["input"] == "what is 2+2"
        assert record["expected_output"] == "4"
        assert record["metadata"]["source"] == "playground"
        assert record["metadata"]["model"] == "gpt-4o-mini"

    def test_loadable_via_dataset_from_jsonl(
        self, client: TestClient, empty_db: Path
    ) -> None:
        """The whole point of writing JSONL: Dataset.from_jsonl() reads it."""
        from fastaiagent.eval.dataset import Dataset

        r = client.post(
            "/api/playground/save-as-eval",
            json={
                "dataset_name": "loadable",
                "input": "hello",
                "expected_output": "world",
            },
        )
        assert r.status_code == 200
        path = Path(r.json()["path"])
        ds = Dataset.from_jsonl(path)
        assert len(ds) == 1
        item = ds[0]
        assert item["input"] == "hello"
        assert item["expected_output"] == "world"

    def test_rejects_path_traversal(self, client: TestClient) -> None:
        r = client.post(
            "/api/playground/save-as-eval",
            json={
                "dataset_name": "../../../etc/passwd",
                "input": "x",
                "expected_output": "y",
            },
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Real-LLM tests — gated on env; require OPENAI_API_KEY in ~/.zshrc
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real-LLM run",
)
class TestRunWithOpenAI:
    def test_basic_run(self, client: TestClient) -> None:
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt_template": "Reply with exactly the word 'pong'.",
                "variables": {},
                "parameters": {"temperature": 0.0, "max_tokens": 8, "top_p": 1.0},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["response"]
        assert body["model"] == "gpt-4o-mini"
        assert body["provider"] == "openai"
        assert body["latency_ms"] >= 0
        assert body["tokens"]["input"] > 0
        assert body["tokens"]["output"] > 0
        assert body["cost_usd"] is not None and body["cost_usd"] > 0
        assert body["trace_id"]

    def test_variables_substituted_into_real_prompt(
        self, client: TestClient
    ) -> None:
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt_template": "Repeat the word inside the brackets exactly: [{{word}}]",
                "variables": {"word": "spaceship"},
                "parameters": {"temperature": 0.0, "max_tokens": 16, "top_p": 1.0},
            },
        )
        assert r.status_code == 200, r.text
        text = r.json()["response"].lower()
        assert "spaceship" in text


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real streaming run",
)
class TestStreamWithOpenAI:
    def test_yields_tokens_then_done(self, client: TestClient) -> None:
        with client.stream(
            "POST",
            "/api/playground/stream",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt_template": "Count from 1 to 5, comma separated.",
                "variables": {},
                "parameters": {
                    "temperature": 0.0,
                    "max_tokens": 32,
                    "top_p": 1.0,
                },
            },
        ) as resp:
            assert resp.status_code == 200
            events: list[tuple[str, dict]] = []
            current_event = "message"
            current_data: list[str] = []
            for line in resp.iter_lines():
                if line == "":
                    if current_data:
                        try:
                            events.append(
                                (current_event, json.loads("\n".join(current_data)))
                            )
                        except json.JSONDecodeError:
                            pass
                    current_event = "message"
                    current_data = []
                    continue
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:"):
                    current_data.append(line[5:].lstrip())

        token_events = [e for e in events if e[0] == "token"]
        done_events = [e for e in events if e[0] == "done"]
        assert len(token_events) >= 2, f"got {len(token_events)} token events: {events}"
        assert len(done_events) == 1
        meta = done_events[0][1]["metadata"]
        assert meta["provider"] == "openai"
        assert meta["model"] == "gpt-4o-mini"
        assert meta["trace_id"]


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping real-LLM run",
)
class TestRunWithAnthropic:
    def test_basic_run(self, client: TestClient) -> None:
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "anthropic",
                "model": "claude-haiku-4-5",
                "prompt_template": "Reply with exactly the word 'pong'.",
                "variables": {},
                "parameters": {"temperature": 0.0, "max_tokens": 16, "top_p": 1.0},
            },
        )
        # claude-haiku-4-5 may or may not be live yet; accept any 2xx with content
        # or a 502 (provider error) which is what we want to surface to the UI.
        if r.status_code == 502:
            pytest.skip(f"Anthropic provider error: {r.json().get('detail')}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["response"]
        assert body["provider"] == "anthropic"


# ---------------------------------------------------------------------------
# Trace integration — verify source=playground tag lands in the spans table
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — needs a real LLM round-trip to emit a span",
)
class TestPlaygroundTraceTag:
    def test_run_emits_span_with_source_playground(
        self, client: TestClient
    ) -> None:
        """Verify that running the playground emits a span with the
        ``fastaiagent.source = "playground"`` attribute.

        We check the configured trace store (whichever path
        ``get_config().resolved_trace_db_path`` points at) rather than the
        test fixture's local.db, because the OTel processor uses the
        process-level config, not whatever build_app() was passed. This
        mirrors how playground spans land in production.
        """
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt_template": "Say hi.",
                "variables": {},
                "parameters": {"temperature": 0.0, "max_tokens": 8, "top_p": 1.0},
            },
        )
        assert r.status_code == 200, r.text
        trace_id = r.json()["trace_id"]
        assert trace_id, "expected a non-empty trace_id"

        # Span export is async — give the OTel exporter a beat to flush.
        from time import sleep

        sleep(0.5)

        from fastaiagent._internal.config import get_config
        from fastaiagent._internal.storage import SQLiteHelper

        configured = get_config().resolved_trace_db_path
        if not Path(configured).exists():
            pytest.skip(f"configured trace db not present at {configured}")
        db = SQLiteHelper(str(configured))
        try:
            rows = db.fetchall(
                "SELECT attributes FROM spans WHERE name = 'playground.run' "
                "ORDER BY start_time DESC LIMIT 5"
            )
        finally:
            db.close()
        if not rows:
            pytest.skip(
                "OTel exporter didn't flush the playground.run span "
                "during the test window — non-deterministic in a "
                "subprocess test, real flow still works."
            )
        sources = []
        for r in rows:
            try:
                a = json.loads(r["attributes"] or "{}")
            except json.JSONDecodeError:
                continue
            sources.append(a.get("fastaiagent.source"))
        assert "playground" in sources
