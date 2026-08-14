"""Example 94: Guardrails + evals from a foreign runtime — compute ≠ emit (1.47.0).

You already have a runtime (LangChain, CrewAI, LlamaIndex, or your own) and you
are not going to rewrite it as a ``fa.Agent``. You still want plane-authored
guardrails enforced at your edge and eval scores attached to the traces you
already emit. That is what the SDK's *primitives* are for.

Two rules make this work:

    1. **Compute ≠ emit.** ``run_guardrail(...)`` returns a verdict and opens no
       span. ``Scorer.score(...)`` returns a score and opens no span. Emission is
       your runtime's job — ``emit_guardrail`` / ``emit_evaluation`` stamp the
       standard OpenInference shape on *your* tracer.
    2. **One exporter per process.** ``fa.connect(export_traces=False)`` connects
       for policy, scorers and prompts without registering a platform span
       exporter and without claiming OpenTelemetry's global tracer provider.

The spans produced here are byte-identical in shape to the ones the SDK's own
agent loop emits, so the platform records the same guardrail rows and trace
scores either way.

This example is deterministic and needs no API key or live plane: it fakes the
``/policy`` payload ``fa.connect()`` would pull, and exports to an in-memory
collector so you can see the exact attributes that go on the wire.
"""

from __future__ import annotations

import asyncio
import json

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import fastaiagent as fa
from fastaiagent.client import _connection
from fastaiagent.guardrail.from_policy import clear_cache

# What GET /public/v1/policy returns: an admin authored a rule that blocks refund
# promises in output, domain-wide.
PLANE_RULE = {
    "id": "gr_refund",
    "name": "no-refund-promises",
    "guardrail_type": "output",
    "validation_mode": "blocking",
    "implementation_type": "regex",
    "config": {"pattern": r"(?i)\b(full refund|issue a refund)\b", "should_match": False},
    "tripwire_message": "Refund promises are not permitted; direct to billing.",
    "on_error": "block",
    "agent_ids": [],
}

QUESTION = "My bill doubled this month. What are my options?"


def foreign_runtime_answer(question: str) -> str:
    """Stand-in for your LangChain/CrewAI/LlamaIndex call. Returns a bad answer."""
    return "Of course — I'll issue a refund to your account right now."


def main() -> int:
    # ── Your runtime's tracer. In a real app this is already set up, exporting
    #    to wherever you send traces (OTLP collector, the platform, Phoenix…).
    collector = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(collector))
    tracer = provider.get_tracer("my.foreign.runtime")

    # ── 1. Connect for policy + scorers ONLY. No second exporter, no global
    #    provider hijack. (With a real key this is the whole connect call.)
    #
    #    fa.connect(api_key="fa_...", target="https://app.fastaiagent.net",
    #               export_traces=False)
    #
    #    Offline stand-in for the /policy pull that connect() performs:
    _connection.policy_cache = {"version": "v1", "guardrail_rules": [PLANE_RULE]}
    clear_cache()

    try:
        # ── 2. PULL — plane-authored guardrails, rebuilt as runnable objects.
        rails = fa.plane_guardrails_for_agent("support-bot")
        print(f"guardrails distributed by the plane: {[g.name for g in rails]}")

        answer = foreign_runtime_answer(QUESTION)
        print(f"runtime answered: {answer!r}\n")

        with tracer.start_as_current_span("agent.run") as root:
            root.set_attribute("openinference.span.kind", "AGENT")
            root.set_attribute("input.value", QUESTION)
            root.set_attribute("output.value", answer)

            for rail in rails:
                # ── 3. COMPUTE — the SDK primitive. Returns a verdict, emits
                #    nothing. It also applies the rule's on_error policy, so a
                #    check that cannot run comes back errored rather than raising.
                result = asyncio.run(fa.run_guardrail(rail, answer))
                outcome = "error" if result.errored else ("pass" if result.passed else "block")

                # ── 4. EMIT — on YOUR tracer, in the standard shape.
                fa.emit_guardrail(
                    tracer,
                    name=rail.name,
                    position=rail.position.value,
                    passed=result.passed,
                    checks=json.dumps([{"name": rail.name, "result": outcome}]),
                    errored=result.errored,
                    message=result.message,
                )
                # Enforcement stays YOURS — the primitive only reports the verdict.
                if not result.passed and rail.blocking:
                    print(f"blocked by {rail.name}: {result.message}")

            # ── 5. SCORE + EMIT. With a live plane this score comes from
            #    fa.Scorer.from_platform("helpfulness").score(...). Judge scales
            #    vary, so normalize to the 0..1 contract before emitting — here a
            #    2-out-of-5 judge verdict.
            fa.emit_evaluation(
                tracer,
                name="helpfulness",
                score=2 / 5,
                label="fail",
                explanation="Promises a refund instead of explaining billing options.",
                annotator_kind="LLM",
            )

        provider.force_flush()
    finally:
        _connection.policy_cache = None
        clear_cache()

    # ── What actually went on the wire.
    print("\nspans emitted on your own tracer:")
    for span in collector.get_finished_spans():
        a = dict(span.attributes or {})
        kind = a.get("openinference.span.kind")
        print(f"\n  {span.name}  [{kind}]  status={span.status.status_code.name}")
        for k, v in sorted(a.items()):
            if k.startswith(("fastaiagent.guardrail.", "evaluation.")):
                print(f"      {k} = {v!r}")

    print(
        "\nThe GUARDRAIL span resolves to a guardrail row on the platform (keyed on\n"
        "fastaiagent.guardrail.name); the EVALUATOR span becomes a per-trace score.\n"
        "Batch/CI scoring stays on evaluate(...) -> EvalResults.publish() — a\n"
        "published run is a dataset measurement, not a per-trace annotation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
