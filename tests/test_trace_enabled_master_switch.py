"""``FASTAIAGENT_TRACE_ENABLED=0`` means *capture nothing at all*.

``docs/security.md`` sells this as the way to keep even local capture from
happening, and ``docs/tracing/index.md`` gives it as the disable recipe — but
until 1.67.0 ``SDKConfig.trace_enabled`` was parsed and read by nothing at all.
With the switch off a span was still built, still written to ``local.db``, and
still carried its payload.

The choke point is ``trace.otel.get_tracer_provider``: with the switch off it
returns OTel's ``NoOpTracerProvider``, so every ``get_tracer()`` in the codebase
mints non-recording spans. Nothing is built, so there is nothing to store and
nothing for any exporter to receive. These tests assert the *outcome* — rows in
``local.db``, bytes on a socket — not the provider's class name.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

import fastaiagent as fa
from fastaiagent._internal.config import reset_config
from fastaiagent.trace import otel

CANARY = "CANARY-a1b2c3-master-switch"


@pytest.fixture
def fresh_tracing(monkeypatch, tmp_path):
    """A clean tracer provider + a private local.db for each case."""
    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    reset_config()
    otel.reset()
    yield db_path
    otel.reset()
    reset_config()


def _span_rows(db_path) -> list[dict]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM spans")]
        except sqlite3.OperationalError:
            return []  # table never created — also "nothing captured"
    finally:
        conn.close()


async def _run_agent() -> object:
    from fastaiagent.llm.client import LLMResponse
    from tests.conftest import MockLLMClient

    reply = LLMResponse(content=f"answer containing {CANARY}", finish_reason="stop")
    agent = fa.Agent(
        name="switch-probe",
        system_prompt=f"You are a probe. {CANARY}",
        llm=MockLLMClient([reply]),
    )
    return await agent.arun(f"question containing {CANARY}")


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", "  false  "])
async def test_switch_off_captures_zero_spans(value, fresh_tracing, monkeypatch):
    monkeypatch.setenv("FASTAIAGENT_TRACE_ENABLED", value)
    reset_config()
    otel.reset()

    await _run_agent()

    rows = _span_rows(fresh_tracing)
    assert rows == [], f"FASTAIAGENT_TRACE_ENABLED={value!r} still captured {len(rows)} span(s)"


@pytest.mark.asyncio
async def test_switch_on_still_captures(fresh_tracing, monkeypatch):
    """The control: the same run with the switch left alone does capture."""
    monkeypatch.delenv("FASTAIAGENT_TRACE_ENABLED", raising=False)
    reset_config()
    otel.reset()

    await _run_agent()

    rows = _span_rows(fresh_tracing)
    assert rows, "tracing was on but nothing was captured — the test is not proving anything"
    assert any(CANARY in (r.get("attributes") or "") for r in rows)


@pytest.mark.asyncio
async def test_switch_off_exports_nothing(fresh_tracing, monkeypatch):
    """No span reaches a registered exporter, and the canary never leaves."""
    monkeypatch.setenv("FASTAIAGENT_TRACE_ENABLED", "false")
    reset_config()
    otel.reset()

    received: list[object] = []

    class _Recorder:
        def export(self, spans):
            received.extend(spans)
            return None

        def shutdown(self):
            return None

        def force_flush(self, timeout_millis=30_000):
            return True

    otel.add_exporter(_Recorder())
    await _run_agent()
    assert received == []


@pytest.mark.asyncio
async def test_a_none_trace_id_degrades_gracefully(fresh_tracing, monkeypatch):
    """Anything reading ``result.trace_id`` must not raise with tracing off.

    A non-recording span has an all-zero, invalid span context, so the id is
    either ``None`` or the 32-zero string depending on the caller. Either is
    fine; raising is not.
    """
    monkeypatch.setenv("FASTAIAGENT_TRACE_ENABLED", "off")
    reset_config()
    otel.reset()

    result = await _run_agent()
    trace_id = result.trace_id
    assert trace_id is None or (isinstance(trace_id, str) and set(trace_id) <= set("0"))
    # The shapes downstream code actually uses.
    assert json.dumps({"trace_id": trace_id})
    assert f"{trace_id}"


@pytest.mark.asyncio
async def test_switch_off_writes_no_attachment_bytes(fresh_tracing, monkeypatch):
    """Attachment bytes are the largest local write; the switch covers them too."""
    from fastaiagent.trace.attachments import save_parts_for_span
    from fastaiagent.trace.storage import TraceStore

    monkeypatch.setenv("FASTAIAGENT_TRACE_ENABLED", "no")
    reset_config()
    otel.reset()

    from fastaiagent.multimodal.image import Image

    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c6360000002000100ffff03000006000557bfabd4000000"
        "0049454e44ae426082"
    )
    store = TraceStore.default()
    saved = save_parts_for_span(
        db=store._db,
        trace_id="0" * 32,
        span_id="0" * 16,
        parts=[Image(data=png, media_type="image/png")],
        role="input",
    )
    assert saved == []
