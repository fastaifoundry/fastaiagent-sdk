"""The guardrail span is unchanged on the wire (1.89.0).

1.89.0 changed what a guardrail check *records locally* (an event row with its
effect, the judged text, the rule). None of that may reach the span the plane
and every OTel exporter receive: same keys, same values, same ``checks``, same
status. This pins each span over blocking × observe-only × every action × every
check state, at the output position (re-ask allowed) and the input position
(re-ask halts), against ``tests/data/guardrail_span_wire_pin.json``.

**The fixture was generated on v1.88.0**, so this test passes on both versions
by construction. To regenerate it (only ever on purpose, from a release whose
wire is the reference)::

    FASTAIAGENT_REGEN_WIRE_PIN=1 pytest tests/test_guardrail_span_wire_pin.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail.executor import execute_guardrails
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType

FIXTURE = Path(__file__).parent / "data" / "guardrail_span_wire_pin.json"

ACTIONS = ["block", "warn", "mask", "override", "reask"]
STATES = ["clean", "fail", "errored_allow", "errored_block"]


class _Collector(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[Any] = []

    def export(self, spans):  # type: ignore[override]
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover
        pass


def _boom(_text: str) -> bool:
    raise RuntimeError("detector down")


def _rule(action: str, blocking: bool, state: str, position: GuardrailPosition) -> Guardrail:
    if state.startswith("errored"):
        return Guardrail(
            name="rule",
            guardrail_type=GuardrailType.code,
            position=position,
            fn=_boom,
            action=action,
            blocking=blocking,
            on_error="allow" if state == "errored_allow" else "block",
        )
    return Guardrail(
        name="rule",
        guardrail_type=GuardrailType.regex,
        position=position,
        config={"pattern": "SECRET", "tripwire_message": "I can't share that."},
        action=action,
        blocking=blocking,
    )


def _case(action: str, blocking: bool, state: str, position: str) -> str:
    return f"{position}/{action}/{'blocking' if blocking else 'observer'}/{state}"


def _snapshot(span: Any) -> dict[str, Any]:
    attrs = {k: v for k, v in dict(span.attributes or {}).items()}
    # JSON-normalise tuples (OTel stores sequence attributes as tuples).
    attrs = json.loads(json.dumps(attrs, default=list, sort_keys=True))
    return {
        "name": span.name,
        "attributes": attrs,
        "status": span.status.status_code.name,
        "description": span.status.description,
    }


def _observe() -> dict[str, Any]:
    from fastaiagent.trace.otel import get_tracer_provider

    collector = _Collector()
    get_tracer_provider().add_span_processor(SimpleSpanProcessor(collector))
    observed: dict[str, Any] = {}
    for position in (GuardrailPosition.output, GuardrailPosition.input):
        for action in ACTIONS:
            for blocking in (True, False):
                for state in STATES:
                    collector.spans.clear()
                    text = "the code is SECRET" if state == "fail" else "all clear"
                    rule = _rule(action, blocking, state, position)

                    async def run(rule: Guardrail = rule, text: str = text) -> None:
                        try:
                            await execute_guardrails(
                                [rule],
                                text,
                                position,
                                allow_reask=position == GuardrailPosition.output,
                            )
                        except GuardrailBlockedError:
                            pass

                    asyncio.run(run())
                    guard = [s for s in collector.spans if s.name.startswith("guardrail.")]
                    assert len(guard) == 1, _case(action, blocking, state, position.value)
                    observed[_case(action, blocking, state, position.value)] = _snapshot(guard[0])
    return observed


@pytest.fixture
def tracing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    for name in ("FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET"):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    otel.reset()
    yield
    otel.reset()
    reset_config()


def test_every_guardrail_span_matches_the_v1_88_wire(tracing: Any) -> None:
    observed = _observe()
    if os.environ.get("FASTAIAGENT_REGEN_WIRE_PIN") == "1":
        FIXTURE.write_text(json.dumps(observed, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"regenerated {FIXTURE.name}")
    pinned = json.loads(FIXTURE.read_text())
    assert set(observed) == set(pinned)
    mismatched = [case for case in pinned if observed[case] != pinned[case]]
    assert not mismatched, (
        f"{len(mismatched)} guardrail span(s) changed on the wire, e.g. {mismatched[0]}:\n"
        f"pinned:   {pinned[mismatched[0]]}\nobserved: {observed[mismatched[0]]}"
    )
