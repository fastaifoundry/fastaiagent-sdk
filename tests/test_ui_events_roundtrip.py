"""End-to-end test for the Events tab on the Trace Detail page.

Uses a real OTel tracer wired to the real ``LocalStorageProcessor``,
opens a real span, raises a real exception inside the span context
manager (which OTel auto-records as an ``exception`` span event), then
reads the span back through the real FastAPI route. Verifies the
``exception.type`` / ``exception.message`` / ``exception.stacktrace``
attributes round-trip end-to-end so the UI has what it needs to render
a useful traceback.

No mocks: real OpenTelemetry, real SQLite, real FastAPI TestClient,
real Python exception.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")
pytest.importorskip("opentelemetry.sdk.trace")

from fastapi.testclient import TestClient  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402

from fastaiagent.trace.storage import LocalStorageProcessor  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


class _BoomError(RuntimeError):
    """Distinct exception class so we can assert exception.type precisely."""


@pytest.fixture
def live_trace_with_exception(tmp_path: Path) -> tuple[Path, str]:
    """Produce a real trace that contains an OTel-recorded exception event.

    Wires a ``TracerProvider`` → ``SimpleSpanProcessor`` → ``LocalStorageProcessor``
    (no exporter batching, so the span is flushed synchronously when the
    context manager exits) against the test's tmp ``local.db``. Then opens
    a span, raises ``_BoomError`` inside it, lets OTel record the exception
    and re-raise.
    """
    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()

    provider = TracerProvider()
    # LocalStorageProcessor implements the OTel SpanProcessor protocol
    # directly (on_start / on_end), so it's registered as-is — no
    # SimpleSpanProcessor wrapper needed.
    processor = LocalStorageProcessor(db_path=str(db_path))
    provider.add_span_processor(processor)  # type: ignore[arg-type]

    tracer = provider.get_tracer("fastaiagent-test")
    trace_id_hex = ""
    with pytest.raises(_BoomError):
        with tracer.start_as_current_span("agent.exploding-bot") as span:
            ctx = span.get_span_context()
            trace_id_hex = format(ctx.trace_id, "032x")
            raise _BoomError("agent ran out of patience")

    # Ensure anything pending is flushed before the HTTP client reads.
    provider.shutdown()
    return db_path, trace_id_hex


@pytest.fixture
def client(live_trace_with_exception, tmp_path: Path) -> TestClient:
    db_path, _ = live_trace_with_exception
    app = build_app(
        db_path=str(db_path),
        auth_path=tmp_path / "auth.json",
        no_auth=True,
    )
    return TestClient(app)


def test_events_round_trip_exception_type_message_stacktrace(
    live_trace_with_exception, client: TestClient
):
    _, trace_id = live_trace_with_exception
    r = client.get(f"/api/traces/{trace_id}/spans")
    assert r.status_code == 200
    tree = r.json()["tree"]
    root = tree["span"]
    events = root["events"]
    assert len(events) == 1, f"expected one exception event, got {events}"

    event = events[0]
    assert event["name"] == "exception"

    attrs = event.get("attributes")
    assert isinstance(attrs, dict), f"attributes missing: {event}"
    exc_type = attrs.get("exception.type") or ""
    # Some Python/OTel versions write bare "_BoomError", others the fully-qualified
    # dotted name. Accept both.
    assert exc_type == "_BoomError" or exc_type.endswith("._BoomError"), attrs
    assert attrs.get("exception.message") == "agent ran out of patience"
    stacktrace = attrs.get("exception.stacktrace") or ""
    assert "_BoomError" in stacktrace, (
        "stacktrace should include the raised exception's class"
    )
    assert "raise _BoomError" in stacktrace, (
        "stacktrace should include the raising source line"
    )


def test_events_empty_for_clean_span(tmp_path: Path):
    """A span that completes without raising should have zero events."""
    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()
    provider = TracerProvider()
    provider.add_span_processor(
        LocalStorageProcessor(db_path=str(db_path))  # type: ignore[arg-type]
    )
    tracer = provider.get_tracer("fastaiagent-test-clean")
    with tracer.start_as_current_span("agent.happy") as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
    provider.shutdown()

    app = build_app(
        db_path=str(db_path),
        auth_path=tmp_path / "auth.json",
        no_auth=True,
    )
    with TestClient(app) as c:
        r = c.get(f"/api/traces/{trace_id}/spans")
    assert r.status_code == 200
    assert r.json()["tree"]["span"]["events"] == []
