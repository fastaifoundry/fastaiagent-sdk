# Concepts & Mental Model

This page explains **what** a trace is in this SDK, **why** the span tree is the
central artifact rather than a log, and **the concept of how** spans get
created, nested, stored, and drained. For the full API see the
[Tracing reference](index.md); for contributor-level detail see
[Tracing architecture](../internals/tracing-architecture.md).

## What it is

Every agent run produces a **trace**: a tree of spans, one per unit of work —
the agent itself, each LLM call, each tool, memory reads and writes, knowledge
base retrievals. It's OpenTelemetry-native, stored locally in SQLite by default,
and exportable anywhere OTel goes.

## Why a tree, not a log

An agent run is a *recursive, non-deterministic* process: the model decides how
many times to loop, which tools to call, in what order, sometimes in parallel. A
flat log tells you what happened; it doesn't tell you **what caused what**. The
tree does — it preserves the causal shape of the run, which is what you need to
answer "why did it call that tool?" or "where did the 8 seconds go?"

That structure is also why the trace becomes the substrate for everything else:
the Local UI renders it, [Replay](../replay/concepts.md) reconstructs an agent
from it, eval [curation](../evaluation/curation.md) turns it into datasets, and
the platform drains it for observability.

## The concept of how

### Nesting is discovered, not declared

No code passes a "parent span" around. A span is opened with
`start_as_current_span`, which pushes it onto OpenTelemetry's context — a
`ContextVar`. Any span opened while that one is active automatically attaches as
its child. Because it's a ContextVar, nesting follows **actual runtime call
flow**, including across `await` boundaries. The tree is a *recording of what
called what*, which is why instrumentation needs no plumbing.

### Flat rows in, tree out

Spans are **not** stored as a tree. Each completed span is written as one flat
row carrying its own `span_id` and `parent_span_id`; the hierarchy is
reconstructed on read by following those links. Two consequences worth knowing:
writes are per-span and **synchronous** — the instant a `with` block exits, that
span is durable on disk, with no batching to lose — and the write is an upsert
keyed by `span_id`, so re-writing a span is idempotent.

!!! info "Verified against a live run"
    A two-turn tool-using agent stored **4 flat rows** with exactly **one
    parentless root** (`agent.trace-probe`). Rebuilding from `parent_span_id`
    reproduced the tree: `agent.trace-probe → [llm.openai.gpt-4.1,
    tool.lookup, llm.openai.gpt-4.1]`.

### One root per run

Each `agent.run()` / chain execution produces exactly one parentless span,
named for its runner (`agent.<name>`, `chain.<name>`, `swarm.<name>`,
`supervisor.<name>`). That root is what makes a trace a *unit of work* you can
list, cost, filter, and replay — it carries the run's identity and provenance.

### Two classification axes

Span **names** (`agent.support`, `llm.openai.gpt-4.1`, `tool.lookup`) are for
humans reading a tree. `fastaiagent.runner.type` is for machines classifying a
span **without parsing its name** — it's what lets exporters and the UI decide
how to treat a span. Chains, swarms, supervisors, tools, memory, and retrieval
set it explicitly; a plain agent root omits it and readers **default to
`agent`**.

!!! info "Verified against a live run"
    In the trace above, `tool.lookup` carried
    `fastaiagent.runner.type="tool"`, while the `agent.*` and `llm.*` spans had
    no explicit `runner.type` — consumers fall back to `"agent"`.

### Attributes: structure vs. content

Two namespaces: the OTel-standard `gen_ai.*` keys (model, temperature, token
usage, finish reason) and the SDK's own `fastaiagent.*` keys (agent, chain,
tool, checkpoint, guardrail, cost, framework).

The design principle for privacy is **the skeleton always survives; only the
flesh is optional.** Setting `FASTAIAGENT_TRACE_PAYLOADS=0` drops free-text
content while structural metadata — provider, model, tool schemas, token counts
— is always kept. A gated trace stays fully useful for monitoring, cost, and
latency, and degrades gracefully for replay.

!!! info "Verified against a live run"
    With `FASTAIAGENT_TRACE_PAYLOADS=0`, the free-text attributes —
    `gen_ai.request.messages`, `gen_ai.response.content`, `agent.input`,
    `agent.output`, and the resolved system prompt — are all absent, while
    `agent.name`, `gen_ai.request.model`, and `fastaiagent.runner.type` remain.
    Structure survives; content doesn't.

    A consequence worth knowing: because the recorded input and system prompt
    are what [Replay](../replay/concepts.md) reconstructs from, a
    payload-gated trace is still fully useful for monitoring, cost, and latency
    but can no longer be replayed faithfully. That's the intended trade-off.

### Redaction is a different knob

Payload gating **drops** whole fields. A `RedactionPolicy` **masks** matching
substrings *within* fields you keep — so you can retain a readable trace with
account numbers or emails starred out. It's off by default (zero overhead when
no policy is installed) and applies at the storage boundary, before the row is
written.

!!! warning "Scope of capture-mode redaction"
    Capture-mode redaction masks what is written to `local.db` — and therefore
    what reaches the platform, since the platform drains *from* SQLite. It does
    **not** currently mask spans handed to an OTel exporter registered via
    `add_exporter()`: that exporter reads the span object directly, which is
    never mutated. If you route traces to an external backend (OTLP, Datadog,
    …), scrub at the exporter layer too.

### `local.db` is the substrate, and the queue

Local SQLite storage isn't a feature you turn on — the storage processor is
welded onto the tracer provider when it's built. Everything else (OTLP,
platform export) is an *additional* sink layered on top; local stays the source
of truth.

That inversion is what makes platform export durable. Spans land locally marked
unsynced; the exporter **ignores the batch handed to it** and instead drains
unsynced rows from SQLite, marking them synced only after a confirmed 2xx. In
other words: **SQLite is the queue and the OTel batch is just a doorbell.** A
platform outage can't lose data — it just leaves rows unsynced until the next
drain.

## Capturing other frameworks

`enable_otel_capture()` does two things at once: it attaches the storage
processor to whatever global tracer provider is active (so import order stops
mattering), and it normalizes foreign conventions — OpenInference, OpenLLMetry
— onto the canonical `gen_ai.*` / `runner.type` keys at write time. The result
is that a LangChain or LlamaIndex run lands in the same store, in the same
shape, as a native one. See [third-party OTel capture](third-party-otel.md).

## Imports

`fastaiagent` exports `TraceStore`, `trace_context`, `enable_otel_capture`,
`Replay`, `RedactionPolicy`, and `set_redaction_policy`. `add_exporter`,
`get_redaction_policy`, `SENSITIVE_ATTR_KEYS`, `TraceData`, and `TraceSummary`
come from `fastaiagent.trace`.

## Next steps

- [Tracing reference](index.md) — `trace_context`, querying `TraceStore`, OTLP export, custom DB path, CLI
- [Third-party OTel capture](third-party-otel.md) — instrument other frameworks into the same store
- [Tracing architecture](../internals/tracing-architecture.md) — contributor-level internals
- [Replay](../replay/concepts.md) — what reads these spans back
- [Security](../security.md) — redaction in the broader privacy picture
