"""FastAPI 0.143 traces every request by default — through fastaiagent's provider (1.84.0).

FastAPI 0.143 sends request spans to the *global* OpenTelemetry tracer provider,
which is fastaiagent's own once the SDK has traced anything. Two defects followed,
both pre-existing and both caught by CI the day FastAPI shipped it:

1. A span the local store could not write raised **inside the HTTP request** —
   tracing broke the app. ``LocalStorageProcessor.on_end`` now never raises.
2. The Local UI recorded each of its own API calls into ``local.db`` as a ``GET``
   trace. The SDK's servers now opt out of FastAPI's built-in telemetry.

No mocks: a real ``TracerProvider`` with the real processor, the real UI app and
``agent serve`` app over ``TestClient``. The self-tracing cases need a FastAPI
that has the setting; CI installs the latest FastAPI, so they run there.
"""

from __future__ import annotations

import inspect
import logging
import sqlite3

import pytest

from fastaiagent.trace.storage import LocalStorageProcessor

fastapi = pytest.importorskip("fastapi")

HAS_FASTAPI_TELEMETRY = "telemetry" in inspect.signature(fastapi.FastAPI.__init__).parameters
needs_telemetry = pytest.mark.skipif(
    not HAS_FASTAPI_TELEMETRY, reason="installed FastAPI has no built-in telemetry (< 0.143)"
)


class TestStorageNeverBreaksTheCaller:
    def test_an_unwritable_store_drops_the_span_and_logs_once(self, tmp_path, caplog) -> None:
        from opentelemetry.sdk.trace import TracerProvider

        # A directory where the database file should be: sqlite cannot open it,
        # the same "unable to open database file" a deleted temp dir gives.
        bad = tmp_path / "not-a-file.db"
        bad.mkdir()
        provider = TracerProvider()
        provider.add_span_processor(LocalStorageProcessor(db_path=str(bad)))
        tracer = provider.get_tracer("t")

        with caplog.at_level(logging.WARNING, logger="fastaiagent.trace.storage"):
            for _ in range(3):
                with tracer.start_as_current_span("user-work"):
                    pass  # ending the span must not raise
        warnings = [r for r in caplog.records if "could not write span" in r.message]
        assert len(warnings) == 1, [r.message for r in caplog.records]
        assert "OperationalError" in warnings[0].message


def _spans(db_path) -> list[str]:
    try:
        return [r[0] for r in sqlite3.connect(db_path).execute("select name from spans")]
    except sqlite3.OperationalError:
        return []


@pytest.fixture
def sdk_owns_the_global_provider(monkeypatch, tmp_path):
    """The situation in any process that has traced: fastaiagent's provider is global."""
    from opentelemetry import trace as otel_trace

    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    otel.reset()
    provider = otel.get_tracer_provider()
    # OTel lets the global be set once per process; point it at this provider
    # for the test so FastAPI's telemetry finds it, and put it back after.
    previous = otel_trace._TRACER_PROVIDER
    otel_trace._TRACER_PROVIDER = provider
    try:
        yield db, provider
    finally:
        otel_trace._TRACER_PROVIDER = previous
        otel.reset()
        reset_config()


@needs_telemetry
class TestSdkServersDoNotTraceThemselves:
    def test_the_local_ui(self, sdk_owns_the_global_provider) -> None:
        from fastapi.testclient import TestClient

        from fastaiagent.ui.server import build_app

        db, provider = sdk_owns_the_global_provider
        client = TestClient(build_app(db_path=str(db), no_auth=True))
        for path in ("/api/traces", "/api/analytics", "/api/traces?limit=5"):
            assert client.get(path).status_code == 200
        provider.force_flush()
        assert _spans(db) == []

    def test_agent_serve(self, sdk_owns_the_global_provider) -> None:
        from fastapi.testclient import TestClient

        from fastaiagent.agent.agent import Agent
        from fastaiagent.cli import agent as agent_cli

        db, provider = sdk_owns_the_global_provider
        client = TestClient(agent_cli._build_app(Agent(name="t", system_prompt="x")))
        assert client.get("/health").status_code == 200
        provider.force_flush()
        assert _spans(db) == []

    def test_a_users_own_app_is_left_alone(self, sdk_owns_the_global_provider) -> None:
        """The opt-out is for the SDK's servers only; the user's app keeps FastAPI's default."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        db, provider = sdk_owns_the_global_provider
        app = FastAPI()

        @app.get("/ping")
        def ping() -> dict:
            return {"ok": True}

        assert TestClient(app).get("/ping").status_code == 200
        provider.force_flush()
        assert _spans(db), "FastAPI's own telemetry should still trace a user app"


def test_kwargs_follow_the_installed_fastapi() -> None:
    from fastaiagent._internal.fastapi_compat import sdk_app_kwargs

    kwargs = sdk_app_kwargs()
    if HAS_FASTAPI_TELEMETRY:
        assert kwargs == {
            "telemetry": {"tracing": False, "metrics": False, "logs": False, "operation_spans": False}
        }
    else:
        assert kwargs == {}  # an older FastAPI is constructed exactly as before
