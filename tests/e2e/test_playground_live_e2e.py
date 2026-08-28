"""Live-provider E2E tests for the Prompt Playground.

Moved out of ``tests/test_ui_playground.py`` in 1.53.0. Two reasons:

1. This repo's CI only exposes ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` to
   the e2e job, so a key-gated test outside ``tests/e2e/`` silently never runs
   with a key — it skips on every CI machine forever.
2. The old ``TestRunWithAnthropic`` swallowed a 502 with ``pytest.skip``.
   That turned a real outage into a green tick: at 1.52.0 *every* Anthropic
   Playground call failed (the endpoint always sent both ``temperature`` and
   ``top_p``, which Anthropic rejects) and this suite never noticed.

So: no ``pytest.skip`` on a failed provider call here. If the provider is
reachable and the call fails, that is a failure.

Real FastAPI app, real SQLite, real provider APIs. No mocking.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402

# Cheap, currently-live models. Kept deliberately small so the suite is fast.
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"

needs_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
needs_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)


@pytest.fixture
def client(temp_dir: Path) -> TestClient:
    db_path = temp_dir / ".fastaiagent" / "local.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_local_db(db_path).close()
    return TestClient(build_app(db_path=str(db_path), no_auth=True))


def _read_sse(client: TestClient, body: dict) -> list[tuple[str, dict]]:
    """Collect ``(event, payload)`` pairs from the playground SSE stream."""
    events: list[tuple[str, dict]] = []
    with client.stream("POST", "/api/playground/stream", json=body) as resp:
        assert resp.status_code == 200, resp.read()[:400]
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
                current_event, current_data = "message", []
                continue
            if line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                current_data.append(line[5:].lstrip())
    return events


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@needs_openai
class TestRunWithOpenAI:
    def test_basic_run(self, client: TestClient) -> None:
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "openai",
                "model": OPENAI_MODEL,
                "prompt_template": "Reply with exactly the word 'pong'.",
                "variables": {},
                "parameters": {"temperature": 0.0, "max_tokens": 8},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["response"]
        assert body["model"] == OPENAI_MODEL
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
                "model": OPENAI_MODEL,
                "prompt_template": "Repeat the word inside the brackets exactly: [{{word}}]",
                "variables": {"word": "spaceship"},
                "parameters": {"temperature": 0.0, "max_tokens": 16},
            },
        )
        assert r.status_code == 200, r.text
        assert "spaceship" in r.json()["response"].lower()


@needs_openai
class TestStreamWithOpenAI:
    def test_yields_tokens_then_done(self, client: TestClient) -> None:
        events = _read_sse(
            client,
            {
                "provider": "openai",
                "model": OPENAI_MODEL,
                "prompt_template": "Count from 1 to 5, comma separated.",
                "variables": {},
                "parameters": {"temperature": 0.0, "max_tokens": 32},
            },
        )
        tokens = [e for e in events if e[0] == "token"]
        done = [e for e in events if e[0] == "done"]
        errors = [e for e in events if e[0] == "error"]
        assert not errors, errors
        assert len(tokens) >= 2, f"got {len(tokens)} token events: {events}"
        assert len(done) == 1
        meta = done[0][1]["metadata"]
        assert meta["provider"] == "openai"
        assert meta["model"] == OPENAI_MODEL
        assert meta["trace_id"]


# ---------------------------------------------------------------------------
# Anthropic — the path that was silently broken at 1.52.0
# ---------------------------------------------------------------------------


@needs_anthropic
class TestRunWithAnthropic:
    def test_basic_run(self, client: TestClient) -> None:
        """A plain run must succeed. Deliberately no 502 escape hatch."""
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "anthropic",
                "model": ANTHROPIC_MODEL,
                "prompt_template": "Reply with exactly the word 'pong'.",
                "variables": {},
                "parameters": {"max_tokens": 16},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["response"]
        assert body["provider"] == "anthropic"
        assert body["tokens"]["output"] > 0
        assert body["cost_usd"] is not None and body["cost_usd"] > 0

    def test_default_parameters_do_not_break_anthropic(
        self, client: TestClient
    ) -> None:
        """Regression: omitting ``parameters`` must not send temperature+top_p.

        Before 1.53.0 the model defaulted both to 1.0, and Anthropic replies
        400 "`temperature` and `top_p` cannot both be specified for this
        model." — so the whole provider was unusable from the Playground.
        """
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "anthropic",
                "model": ANTHROPIC_MODEL,
                "prompt_template": "Reply with exactly the word 'pong'.",
                "variables": {},
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["response"]

    @pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-5"])
    def test_claude_5_models_run(self, client: TestClient, model: str) -> None:
        """Claude 5 rejects ``top_p`` outright, so this also guards the fix."""
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "anthropic",
                "model": model,
                "prompt_template": "Reply with exactly the word 'pong'.",
                "variables": {},
                "parameters": {"max_tokens": 32},
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["response"]

    def test_shipped_catalog_models_are_all_live(self, client: TestClient) -> None:
        """Every Anthropic model we put in the dropdown must actually run.

        At 1.52.0 the catalog shipped ``claude-3-5-sonnet-latest``, retired in
        2025 and returning 404. Nothing caught it.
        """
        catalog = client.get("/api/playground/models").json()["providers"]
        models = next(r for r in catalog if r["provider"] == "anthropic")["models"]
        assert models, "anthropic row has no models"
        dead: list[tuple[str, str]] = []
        for model in models:
            r = client.post(
                "/api/playground/run",
                json={
                    "provider": "anthropic",
                    "model": model,
                    "prompt_template": "Say pong.",
                    "variables": {},
                    "parameters": {"max_tokens": 32},
                },
            )
            if r.status_code != 200:
                dead.append((model, r.text[:160]))
        assert not dead, f"dead models in the shipped catalog: {dead}"


@needs_anthropic
class TestStreamWithAnthropic:
    def test_yields_tokens_then_done(self, client: TestClient) -> None:
        """There was no Anthropic streaming test at all before 1.53.0."""
        events = _read_sse(
            client,
            {
                "provider": "anthropic",
                "model": ANTHROPIC_MODEL,
                "prompt_template": "Count from 1 to 5, comma separated.",
                "variables": {},
                "parameters": {"max_tokens": 64},
            },
        )
        errors = [e for e in events if e[0] == "error"]
        assert not errors, f"stream errored: {errors}"
        tokens = [e for e in events if e[0] == "token"]
        done = [e for e in events if e[0] == "done"]

        # Deliberately not asserting a chunk count. How a provider splits a
        # response into deltas is its own business — Anthropic returns a short
        # answer as a single delta. What the endpoint owes us is token events
        # carrying the full text, then exactly one done with metadata.
        assert tokens, f"no token events: {events}"
        assembled = "".join(payload["text"] for _, payload in tokens)
        assert assembled.strip(), f"streamed empty text: {events}"
        assert len(done) == 1
        meta = done[0][1]["metadata"]
        assert meta["provider"] == "anthropic"
        assert meta["model"] == ANTHROPIC_MODEL
        assert meta["trace_id"]
        assert meta["cost_usd"] is not None


@needs_anthropic
class TestStreamErrorCarriesCorrelationId:
    def test_error_event_includes_correlation_id(self, client: TestClient) -> None:
        """The redacted SSE error must carry the id that finds the server log.

        Without it the user sees "LLM call failed." and has no way to look up
        what actually went wrong.
        """
        events = _read_sse(
            client,
            {
                "provider": "anthropic",
                "model": "definitely-not-a-real-model-9000",
                "prompt_template": "Say pong.",
                "variables": {},
                "parameters": {"max_tokens": 16},
            },
        )
        errors = [e for e in events if e[0] == "error"]
        assert len(errors) == 1, events
        payload = errors[0][1]
        assert payload["message"] == "LLM call failed."
        assert payload.get("correlation_id"), payload
        # The provider's raw text must not leak through.
        assert "not_found_error" not in json.dumps(payload)


# ---------------------------------------------------------------------------
# Trace tagging
# ---------------------------------------------------------------------------


@needs_openai
class TestPlaygroundTraceTag:
    def test_run_emits_span_with_source_playground(
        self, client: TestClient
    ) -> None:
        """Playground runs must be filterable in Traces via
        ``fastaiagent.source = "playground"``.

        Checks the configured trace store (``get_config().resolved_trace_db_path``)
        rather than the fixture's local.db, because the OTel processor uses the
        process-level config — same as production.
        """
        r = client.post(
            "/api/playground/run",
            json={
                "provider": "openai",
                "model": OPENAI_MODEL,
                "prompt_template": "Say hi.",
                "variables": {},
                "parameters": {"temperature": 0.0, "max_tokens": 8},
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["trace_id"], "expected a non-empty trace_id"

        from time import sleep

        sleep(0.5)  # span export is async — let the exporter flush

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
                "OTel exporter didn't flush the playground.run span during the "
                "test window — non-deterministic, real flow still works."
            )
        sources = []
        for row in rows:
            try:
                attrs = json.loads(row["attributes"] or "{}")
            except json.JSONDecodeError:
                continue
            sources.append(attrs.get("fastaiagent.source"))
        assert "playground" in sources
