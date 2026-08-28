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
# Sampling parameters — regression for the 1.52.0 Anthropic outage
# ---------------------------------------------------------------------------


class TestParameterDefaults:
    """``temperature`` and ``top_p`` must not be sent unless asked for.

    At 1.52.0 both defaulted to 1.0, so every request carried both. Anthropic
    replies 400 "`temperature` and `top_p` cannot both be specified for this
    model", and Claude 5 rejects ``top_p`` on its own — which made the whole
    provider unusable from the Playground. The live proof lives in
    ``tests/e2e/test_playground_live_e2e.py``; this is the fast unit guard.
    """

    def test_unset_params_are_omitted_from_client_kwargs(self) -> None:
        params = playground_route.PlaygroundParameters()
        assert params.temperature is None
        assert params.top_p is None
        assert params.client_kwargs() == {"max_tokens": 1024}

    def test_explicit_params_are_forwarded(self) -> None:
        params = playground_route.PlaygroundParameters(temperature=0.2, top_p=0.9)
        assert params.client_kwargs() == {
            "temperature": 0.2,
            "max_tokens": 1024,
            "top_p": 0.9,
        }

    def test_zero_temperature_is_forwarded_not_dropped(self) -> None:
        """0.0 is falsy but meaningful — it must survive the None filter."""
        params = playground_route.PlaygroundParameters(temperature=0.0)
        assert params.client_kwargs()["temperature"] == 0.0

    def test_null_max_tokens_is_omitted(self) -> None:
        params = playground_route.PlaygroundParameters(max_tokens=None)
        assert "max_tokens" not in params.client_kwargs()

    def test_request_without_parameters_block_sends_neither(self) -> None:
        body = playground_route.PlaygroundRunRequest(
            provider="anthropic", model="claude-opus-5", prompt_template="hi"
        )
        assert body.parameters.client_kwargs() == {"max_tokens": 1024}


# ---------------------------------------------------------------------------
# Endpoint + per-request credential (AI-gateway path)
# ---------------------------------------------------------------------------


class TestConnectionOverrides:
    """``base_url`` / ``api_key`` let the Playground reach a gateway.

    Org LLM endpoints usually sit behind an OpenAI-compatible gateway on a
    private URL with a bearer token, and those tokens are often short-lived —
    so they can't come from an env var read once at server start.
    """

    def test_unset_overrides_are_omitted(self) -> None:
        body = playground_route.PlaygroundRunRequest(
            provider="openai", model="gpt-4o-mini", prompt_template="hi"
        )
        assert body.connection_kwargs() == {}

    def test_set_overrides_are_forwarded(self) -> None:
        body = playground_route.PlaygroundRunRequest(
            provider="custom",
            model="house-7b",
            prompt_template="hi",
            base_url="https://ai-gateway.internal/v1",
            api_key="tok-123",
        )
        assert body.connection_kwargs() == {
            "base_url": "https://ai-gateway.internal/v1",
            "api_key": "tok-123",
        }

    @pytest.mark.parametrize(
        "bad",
        ["file:///etc/passwd", "gopher://x", "not-a-url", "javascript:alert(1)", "//x"],
    )
    def test_non_http_base_url_rejected(self, client: TestClient, bad: str) -> None:
        """The server makes the outbound call, so the scheme is ours to police."""
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "custom",
                "model": "m",
                "prompt_template": "hi",
                "variables": {},
                "base_url": bad,
                "api_key": "x",
            },
        )
        assert r.status_code == 422, r.text

    def test_whitespace_only_credential_is_treated_as_absent(self) -> None:
        body = playground_route.PlaygroundRunRequest(
            provider="openai", model="m", prompt_template="hi", api_key="   "
        )
        assert body.api_key is None
        assert body.connection_kwargs() == {}

    def test_request_token_satisfies_the_key_gate(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no env key, a per-run token must still get past the 400 gate.

        It should fail later at the provider (502), not up front (400).
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        body = {
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
            "prompt_template": "hi",
            "variables": {},
            "base_url": "https://nonexistent-gateway.invalid/v1",
            "api_key": "tok-abc",
            "parameters": {"max_tokens": 8},
        }
        r = client.post("/api/playground/run", json=body)
        assert r.status_code != 400, r.text

    def test_missing_key_and_no_token_still_400s(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "anthropic",
                "model": "claude-haiku-4-5",
                "prompt_template": "hi",
                "variables": {},
            },
        )
        assert r.status_code == 400
        assert "ANTHROPIC_API_KEY" in r.text

    def test_credential_never_echoed_in_the_response(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        secret = "sk-CANARY-should-not-appear"
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "anthropic",
                "model": "claude-haiku-4-5",
                "prompt_template": "hi",
                "variables": {},
                "base_url": "https://nonexistent-gateway.invalid/v1",
                "api_key": secret,
                "parameters": {"max_tokens": 8},
            },
        )
        assert secret not in r.text


class TestLocalProvidersAreNotKeyGated:
    """You run these servers, so there's usually nothing to authenticate to.

    lmstudio and vllm declare env vars on their presets, which made them show
    as disabled in the UI — you had to invent a dummy key to use a local
    server that wanted none.
    """

    @pytest.mark.parametrize("provider", ["ollama", "lmstudio", "vllm"])
    def test_local_provider_is_selectable_without_a_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, provider: str
    ) -> None:
        monkeypatch.delenv("LMSTUDIO_API_KEY", raising=False)
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
        rows = client.get("/api/playground/models").json()["providers"]
        row = next(r for r in rows if r["provider"] == provider)
        assert row["has_key"] is True, f"{provider} should not be key-gated"

    def test_custom_endpoint_provider_is_offered(self, client: TestClient) -> None:
        """`custom` is the point-at-your-own-gateway provider; it must appear."""
        rows = client.get("/api/playground/models").json()["providers"]
        assert "custom" in {r["provider"] for r in rows}


class TestEnvKeyNeverFollowsACustomEndpoint:
    """The server's own key must not be sent to a user-supplied endpoint.

    ``provider`` picks the wire format; ``base_url`` picks the destination.
    LLMClient falls back to the provider's env var when no key is passed —
    right when base_url is the provider's own URL, and a key-exfiltration path
    the moment it isn't. Verified live: before this guard, an endpoint override
    with an empty token box sent OPENAI_API_KEY to whatever host was typed.

    Passing an empty key doesn't help — LLMClient does
    ``self.api_key or os.environ.get(...)`` and "" is falsy — so the request
    is refused instead.
    """

    def test_endpoint_without_token_is_refused_when_env_key_exists(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "openai",
                "model": "m",
                "prompt_template": "hi",
                "variables": {},
                "base_url": "https://someone-elses-host.example/v1",
            },
        )
        assert r.status_code == 400, r.text
        detail = r.json()["detail"]
        # The message must name the variable at risk and the destination.
        assert "OPENAI_API_KEY" in detail
        assert "someone-elses-host.example" in detail
        # ...and must not contain the key itself.
        assert "sk-real-key" not in r.text

    def test_endpoint_with_token_is_allowed(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "openai",
                "model": "m",
                "prompt_template": "hi",
                "variables": {},
                "base_url": "https://gateway.internal/v1",
                "api_key": "gw-token",
                "parameters": {"max_tokens": 8},
            },
        )
        # Gets past the gate; fails later at the unreachable host.
        assert r.status_code != 400, r.text

    def test_endpoint_without_token_is_fine_when_nothing_could_leak(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A local server on a custom port needs no token and has no key to leak."""
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "vllm",
                "model": "local-model",
                "prompt_template": "hi",
                "variables": {},
                "base_url": "http://127.0.0.1:9999/v1",
                "parameters": {"max_tokens": 8},
            },
        )
        assert r.status_code != 400, r.text

    def test_no_endpoint_still_uses_the_env_key_normally(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard must not break the ordinary path."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "openai",
                "model": "m",
                "prompt_template": "hi",
                "variables": {},
                "parameters": {"max_tokens": 8},
            },
        )
        assert r.status_code != 400, r.text
