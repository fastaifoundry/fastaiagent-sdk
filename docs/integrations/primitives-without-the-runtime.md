# Using guardrails and evals without the agent runtime

You already have a runtime. It's a LangChain graph, a CrewAI crew, a LlamaIndex
query engine, or something you wrote yourself — and you're not going to rewrite
it as a `fa.Agent`. You still want the two things the platform gives you:
guardrails **authored centrally** and enforced at your edge, and eval scores
attached to the traces you already emit.

That's what this page is for. FastAIAgent's guardrail and eval logic is usable
as a **library of primitives**, independent of `fa.Agent`. This is a supported,
public surface — not a reach into internals.

## The two rules

Everything on this page follows from these:

!!! abstract "Compute ≠ emit"
    A primitive **returns a result and emits nothing**. `run_guardrail(...)`
    gives you a `GuardrailResult`; `Scorer.score(...)` gives you a
    `ScorerResult`. Neither opens a span. Emission is your runtime's job,
    through the exporter it already owns.

    The SDK's own agent loop couples the two (`execute_guardrails` computes
    *and* emits) because there it owns both halves. You get the halves
    separately.

!!! abstract "One exporter per process"
    Two exporters means duplicate spans, split traces, or both. Your runtime
    already has one. So `connect(export_traces=False)` — the SDK connects for
    policy, scorers and prompts, but registers **no** span exporter and does
    **not** claim OpenTelemetry's global tracer provider.

## Connect without taking over tracing

```python
import fastaiagent as fa

fa.connect(
    api_key="fa_...",
    target="https://app.fastaiagent.net",
    export_traces=False,   # ← no platform span exporter, no global provider claim
)
```

Everything except trace export still works: the governance policy is pulled and
cached, `Scorer.from_platform` resolves, prompts resolve. What doesn't happen is
a second `PlatformSpanExporter` on your provider.

!!! warning "Call it before your first span"
    OpenTelemetry's global provider is *first-wins*. If your framework sets its
    provider first (the normal case — instrument, then connect), you're fine
    either way. But once a provider claims the global slot it can't be
    unclaimed, so put `connect(export_traces=False)` early.

## Guardrails: pull → compute → emit

Guardrails authored on the plane arrive through the cached `/policy` document.
`plane_guardrails_for_agent` reconstructs them as runnable `Guardrail` objects —
`regex`, `schema`, `classifier` and `llm_judge` rules rebuild from their config;
a `code` rule can't (its logic is a server-side callable) and is skipped.

```python
import json
import fastaiagent as fa
from opentelemetry import trace

tracer = trace.get_tracer("my.runtime")   # YOUR tracer, YOUR exporter

# 1. Pull — the guardrails the plane distributed to this agent.
guardrails = fa.plane_guardrails_for_agent("support-bot")

async def check(text: str, position: str = "output") -> bool:
    for rail in guardrails:
        if rail.position.value != position:
            continue

        # 2. Compute — returns a verdict, emits nothing.
        result = await fa.run_guardrail(rail, text)

        # 3. Emit — one child span on your tracer, standard wire shape.
        outcome = "error" if result.errored else ("pass" if result.passed else "block")
        fa.emit_guardrail(
            tracer,
            name=rail.name,
            position=position,
            passed=result.passed,
            checks=json.dumps([{"name": rail.name, "result": outcome}]),
            errored=result.errored,
            message=result.message,
        )
        if not result.passed and rail.blocking:
            return False
    return True
```

The span this produces is byte-for-byte the shape the SDK's own runtime emits,
so the plane writes the same `guardrail_executions` row either way. Enforcement
— what you *do* with a block — stays yours: the primitive tells you the verdict,
your runtime decides whether to stop.

`run_guardrail` also enforces the guardrail's `on_error` policy for you: if a
check can't run (an `llm_judge` whose model call fails), you get a result with
`errored=True` and `passed` set per `allow`/`block`, never an exception.

## Evals: score → emit

A per-trace score is an `EVALUATOR` span carrying the OpenInference
`evaluation.*` attributes. `Scorer.from_platform` pulls a judge configured on
the plane; `emit_evaluation` attaches its score to the trace you just produced.

```python
scorer = fa.Scorer.from_platform("correctness")

result = scorer.score(input=user_msg, output=answer)

fa.emit_evaluation(
    tracer,
    name="correctness",
    score=result.score,            # MUST be 0..1
    label="pass" if result.passed else "fail",
    explanation=result.reason,
    annotator_kind="LLM",          # or "CODE" / "HUMAN"
)
```

!!! danger "`evaluation.score` is a 0..1 scale"
    A judge on a 1..5 scale must be normalized by you — `score / 5`. An
    out-of-range value is clamped to `[0, 1]` and logged as a warning, so a
    scale mistake shows up in your logs rather than silently landing as a
    perfect score.

## Emitting onto a span you already have

`emit_guardrail` / `emit_evaluation` open their own child span. When you'd
rather stamp an existing span — annotating a span your framework created, or
building the span yourself — use the attribute setters directly:

```python
from fastaiagent import set_guardrail_attributes, set_evaluation_attributes

with tracer.start_as_current_span("my.check") as span:
    set_guardrail_attributes(
        span, name="no_pii", position="output", passed=True,
        checks='[{"name": "no_pii", "result": "pass"}]',
    )
```

Same attributes, your span lifecycle, your status codes.

## The wire contract

Both emitters produce **OpenInference**-classified spans. That's what makes an
SDK-run agent and a borrowed-primitive runtime indistinguishable downstream.

| Span | Classifier | Payload |
|------|-----------|---------|
| Guardrail | `openinference.span.kind = "GUARDRAIL"` | `fastaiagent.guardrail.{name,position,passed,errored,checks}` |
| Inline eval | `openinference.span.kind = "EVALUATOR"` | `evaluation.{name,score,label,explanation,annotator_kind}` |

OpenInference standardizes the *kind*, not guardrail outcome fields — there is
no ecosystem standard for those — so `fastaiagent.guardrail.*` is our documented
convention riding under the standard kind. `fastaiagent.guardrail.name` is how
the plane resolves the span to a guardrail row, so always send it.

Guardrail spans also still carry the legacy `span_type="guardrail"` marker
alongside the standard kind. That dual-write is transitional, for plane
deployments predating the OpenInference reader; don't build on it.

Transport doesn't matter. Your OTLP exporter and the SDK's exporter converge at
the same ingest path on the plane. The rule is one exporter per *process*, not
one transport across the fleet.

## What this is not

- **Not a replacement for batch eval.** An `EVALUATOR` span is a *per-trace*
  score. Dataset / CI scoring stays on `evaluate(...)` →
  [`EvalResults.publish()`](../evaluation/index.md). Those are the only two
  granularities, and publish-a-run is not a per-trace annotation.
- **Not approval-gated governance.** Reconstructed policy guardrails are
  pass/fail checks. Human-in-the-loop tool approval needs the agent runtime —
  see [Managed governance](../guardrails/managed-governance.md).
- **Not a way to get two exporters.** If you find yourself wanting the platform
  exporter *and* your framework's, you want
  [`enable_otel_capture()`](../tracing/third-party-otel.md) on the SDK's runtime
  instead — one exporter, both span sources.

## Public surface

Everything above is exported from the top-level package and covered by the
stability commitment:

| Symbol | What it does |
|--------|--------------|
| `fa.run_guardrail(guardrail, data)` | Compute one guardrail verdict. Async. Applies `on_error`. Emits nothing. |
| `fa.plane_guardrails_for_agent(agent_id)` | Rebuild the plane-authored guardrails scoped to an agent. |
| `fa.guardrail_from_policy_rule(rule)` | Rebuild a single `/policy` rule; `None` if not locally enforceable. |
| `fa.emit_guardrail(tracer, ...)` | Open + stamp + close a `GUARDRAIL` span on your tracer. |
| `fa.emit_evaluation(tracer, ...)` | Open + stamp + close an `EVALUATOR` span on your tracer. |
| `fa.set_guardrail_attributes(span, ...)` | Stamp guardrail attributes on a span you own. |
| `fa.set_evaluation_attributes(span, ...)` | Stamp `evaluation.*` attributes on a span you own. |

## Next steps

- [Managed governance](../guardrails/managed-governance.md) — how guardrails get authored on the plane in the first place
- [Guardrails — concepts](../guardrails/concepts.md) — positions, types, and the outcome attributes
- [Evaluation — concepts](../evaluation/concepts.md) — inline score vs batch run
- [Capture any OTel / OpenInference framework](../tracing/third-party-otel.md) — the other direction: SDK runtime, foreign spans
