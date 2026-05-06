"""Extract durable facts from completed traces — offline, LLM-driven.

The extractor is the **first layer** of Harrison Chase's "continual
learning" framing applied to fastaiagent-sdk: it reads a finished trace
out of ``local.db``, identifies durable user / project / agent facts,
and hands them to :class:`fastaiagent.learn.store.MemoryStore` for
persistence.

What we extract (v1, memory-only):

  * **user facts** — preferences, constraints, recurring asks
    ("the user prefers reports under 800 words")
  * **project facts** — domain context, naming conventions, decisions
    ("this team treats Postgres as source of truth")
  * **agent facts** — operational lessons learned by an agent across
    runs ("the deep-research writer should always include citations
    for token-cost claims")

What we DON'T extract (deferred to PR C / future work):

  * **skills** — reusable mini-procedures
  * **prompt deltas** — Meta-Harness-style harness mutations
  * **PII** — we explicitly avoid extracting names, emails, phone
    numbers; the prompt instructs the model to skip them. Privacy is
    further gated on the CLI: ``user`` / ``project`` scopes require an
    explicit opt-in flag (see :mod:`fastaiagent.learn.cli`).

Conflict resolution is a **separate** concern handled by the store —
this module's job is to produce candidates.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from fastaiagent.learn.store import Fact, MemoryStore, Scope
from fastaiagent.llm.client import LLMClient
from fastaiagent.llm.message import UserMessage
from fastaiagent.trace.storage import TraceData, TraceStore

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class ExtractionResult:
    """Outcome of extracting facts from a single trace."""

    trace_id: str
    candidates: list[Fact]
    written_ids: list[int]  # row ids actually inserted (or matched dedup)
    error: str | None = None


# ─── Prompt template ─────────────────────────────────────────────────────────


_EXTRACTION_PROMPT = """You are extracting durable facts from a completed agent trace.

Scope: {scope}
Scope id: {scope_id}

Below is the trace summary — span names, tool calls, and key text content.
Extract any durable {scope}-level facts that would be useful to inject into
future agent runs covering similar tasks.

Return a JSON array of short fact strings (each under 160 characters,
at most {max_facts} items). Each fact must be:
  * Specific and verifiable (not opinions, not speculation)
  * Generalizable beyond this single trace
  * Relevant to the {scope} scope
  * Free of PII (no names, emails, phone numbers, addresses)

Return [] if nothing durable can be extracted. Do NOT explain — just the
JSON array.

Trace:
{trace_text}

JSON:"""


def _summarize_trace_for_extraction(trace: TraceData, max_chars: int = 12000) -> str:
    """Compress a trace into a text summary the LLM can read.

    We surface span names + the text-bearing GenAI / research attributes,
    skipping low-signal noise. Output is capped at ``max_chars`` so a
    pathologically long trace can't blow the extractor's context window.
    """
    chunks: list[str] = []
    for span in trace.spans:
        attrs = span.attributes or {}
        # Capture the highest-signal text from each span.
        prompt_text = attrs.get("gen_ai.request.messages") or attrs.get("gen_ai.prompt")
        response_text = attrs.get("gen_ai.response.content") or attrs.get(
            "gen_ai.response.text"
        )
        # Research-template structured payloads (from PR A).
        brief = attrs.get("fastaiagent.research.brief")
        findings = attrs.get("fastaiagent.research.findings")
        topic = attrs.get("fastaiagent.research.topic")

        line = f"[{span.name}]"
        if topic:
            line += f" topic={topic!r}"
        if brief:
            line += f" brief={str(brief)[:400]}"
        if findings:
            line += f" findings={str(findings)[:400]}"
        if prompt_text:
            line += f" prompt={str(prompt_text)[:400]}"
        if response_text:
            line += f" response={str(response_text)[:400]}"
        chunks.append(line)

        running = "\n".join(chunks)
        if len(running) > max_chars:
            return running[:max_chars] + "\n\n[...truncated]"
    return "\n".join(chunks)


# ─── Public API ──────────────────────────────────────────────────────────────


def extract_facts_from_trace(
    trace: TraceData,
    *,
    llm: LLMClient,
    scope: Scope,
    scope_id: str = "",
    project_id: str = "",
    max_facts: int = 10,
) -> list[Fact]:
    """LLM-driven fact extraction from a single trace.

    Returns the candidate ``Fact`` objects (not yet persisted). The caller
    decides whether to write them via :class:`MemoryStore`. The trace's
    ``trace_id`` is recorded as ``source_trace_id`` on each fact.

    On extraction failure (LLM error, malformed JSON), returns ``[]`` and
    logs a warning. We never raise — bad traces shouldn't kill the loop.
    """
    if scope not in ("user", "project", "agent"):
        raise ValueError(f"scope must be one of user|project|agent, got {scope!r}")

    trace_text = _summarize_trace_for_extraction(trace)
    if not trace_text.strip():
        return []

    prompt = _EXTRACTION_PROMPT.format(
        scope=scope,
        scope_id=scope_id or "(unset)",
        max_facts=max_facts,
        trace_text=trace_text,
    )

    try:
        response = llm.complete([UserMessage(prompt)])
    except Exception as err:
        _log.warning("Fact extraction LLM call failed for trace %s: %s", trace.trace_id, err)
        return []

    text = (response.content or "").strip()
    if not text:
        return []

    # Tolerate code fences the LLM may wrap the JSON in.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        _log.warning("Fact extraction returned non-JSON for trace %s", trace.trace_id)
        return []

    if not isinstance(data, list):
        return []

    now = time.time()
    facts: list[Fact] = []
    for item in data[:max_facts]:
        text_item = str(item).strip() if item else ""
        if not text_item or len(text_item) > 240:
            continue
        facts.append(
            Fact(
                scope=scope,
                scope_id=scope_id,
                fact=text_item,
                source_trace_id=trace.trace_id,
                confidence=1.0,
                created_at=now,
                project_id=project_id,
            )
        )
    return facts


def extract_and_store(
    trace: TraceData,
    *,
    llm: LLMClient,
    store: MemoryStore,
    scope: Scope,
    scope_id: str = "",
    project_id: str = "",
    max_facts: int = 10,
    dry_run: bool = False,
) -> ExtractionResult:
    """Convenience: extract from one trace + persist via the store.

    With ``dry_run=True``, candidates are returned but nothing is written.
    """
    candidates = extract_facts_from_trace(
        trace,
        llm=llm,
        scope=scope,
        scope_id=scope_id,
        project_id=project_id,
        max_facts=max_facts,
    )
    written: list[int] = []
    if not dry_run and candidates:
        written = store.add_many(candidates)
    return ExtractionResult(
        trace_id=trace.trace_id,
        candidates=candidates,
        written_ids=written,
    )


def run_extraction(
    *,
    llm: LLMClient,
    store: MemoryStore,
    scope: Scope,
    scope_id: str = "",
    project_id: str = "",
    last_hours: int = 24,
    max_facts_per_trace: int = 10,
    dry_run: bool = False,
    trace_store: TraceStore | None = None,
) -> list[ExtractionResult]:
    """Run extraction over every trace in the configured time window.

    The default ``trace_store`` reads from the same ``local.db`` the SDK
    writes to. Override for tests or to point at a different store.
    """
    ts = trace_store if trace_store is not None else TraceStore()
    summaries = ts.list_traces(last_hours=last_hours)

    results: list[ExtractionResult] = []
    for summary in summaries:
        try:
            trace = ts.get_trace(summary.trace_id)
        except Exception as err:
            _log.warning("Failed to load trace %s: %s", summary.trace_id, err)
            results.append(
                ExtractionResult(
                    trace_id=summary.trace_id,
                    candidates=[],
                    written_ids=[],
                    error=str(err),
                )
            )
            continue
        result = extract_and_store(
            trace,
            llm=llm,
            store=store,
            scope=scope,
            scope_id=scope_id,
            project_id=project_id,
            max_facts=max_facts_per_trace,
            dry_run=dry_run,
        )
        results.append(result)
    return results
