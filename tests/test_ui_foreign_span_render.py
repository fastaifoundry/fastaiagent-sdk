"""UI-layer test: a normalized foreign span renders richly (no mocks).

Emits a real OpenInference-style root span through a real ``TracerProvider`` +
``LocalStorageProcessor`` with capture normalization ON, into a real SQLite DB,
then serves that DB through the real FastAPI app and asserts the trace-detail
endpoint returns model, tokens, cost, framework badge, runner type, and the
prompt/response content the Local UI renders.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402

from fastaiagent.trace.storage import (  # noqa: E402
    LocalStorageProcessor,
    TraceStore,
    set_normalize_enabled,
)
from fastaiagent.ui.server import build_app  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_flag():
    set_normalize_enabled(False)
    yield
    set_normalize_enabled(False)


@pytest.fixture
def seeded_foreign_db(tmp_path: Path) -> tuple[str, str]:
    """Return (db_path, trace_id) for a normalized OpenInference root span."""
    db_path = str(tmp_path / "local.db")
    set_normalize_enabled(True)
    provider = TracerProvider()
    processor = LocalStorageProcessor(db_path=db_path)
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("openinference.instrumentation.openai")
    with tracer.start_as_current_span("ChatOpenAI") as span:
        span.set_attribute("llm.model_name", "gpt-4o-mini")
        span.set_attribute("llm.token_count.prompt", 100)
        span.set_attribute("llm.token_count.completion", 50)
        span.set_attribute("input.value", "Summarize the quarterly report.")
        span.set_attribute("output.value", "Revenue rose 12% QoQ.")
        span.set_attribute("openinference.span.kind", "LLM")
    processor.shutdown()

    store = TraceStore(db_path=db_path)
    try:
        trace_id = store.list_traces()[0].trace_id
    finally:
        store.close()
    return db_path, trace_id


@pytest.fixture
def client(seeded_foreign_db: tuple[str, str]) -> TestClient:
    db_path, _ = seeded_foreign_db
    app = build_app(db_path=db_path, no_auth=True)
    return TestClient(app)


def test_trace_detail_renders_rich_foreign_span(
    client: TestClient, seeded_foreign_db: tuple[str, str]
) -> None:
    _, trace_id = seeded_foreign_db
    r = client.get(f"/api/traces/{trace_id}")
    assert r.status_code == 200
    body = r.json()

    # Tokens rolled up from the normalized gen_ai.usage.* keys.
    assert body["total_tokens"] == 150
    # Cost computed from gen_ai.request.model + tokens (gpt-4o-mini priced).
    assert body["total_cost_usd"] is not None and body["total_cost_usd"] > 0
    # Framework badge derived from the instrumentation scope, root span only.
    assert body["framework"] == "openai"
    # Span-kind classification from openinference.span.kind.
    assert body["runner_type"] == "llm"

    # The span carries the canonical keys the IO panels read.
    attrs = body["spans"][0]["attributes"]
    assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
    # FTS / search keys.
    assert attrs["gen_ai.prompt"] == "Summarize the quarterly report."
    assert attrs["gen_ai.completion"] == "Revenue rose 12% QoQ."
    # UI span IO-panel keys (SpanInspector reads these).
    assert attrs["gen_ai.request.messages"] == "Summarize the quarterly report."
    assert attrs["gen_ai.response.content"] == "Revenue rose 12% QoQ."
    # Originals are still present (non-destructive enrichment).
    assert attrs["llm.model_name"] == "gpt-4o-mini"


def test_trace_is_searchable_by_content(
    client: TestClient, seeded_foreign_db: tuple[str, str]
) -> None:
    # FTS index was fed from the normalized gen_ai.prompt/completion, so the
    # free-text search box finds the foreign span by its content.
    r = client.get("/api/traces", params={"q": "quarterly"})
    assert r.status_code == 200
    rows = r.json()["rows"]
    _, trace_id = seeded_foreign_db
    assert any(row["trace_id"] == trace_id for row in rows)
