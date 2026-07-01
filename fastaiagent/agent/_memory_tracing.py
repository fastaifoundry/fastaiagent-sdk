"""OTel tracing helpers for agent memory read/write.

Mirrors :mod:`fastaiagent.kb._tracing` so memory observability reads with the
same mental model as KB retrieval. Two entry points wrap the memory call sites
in :class:`fastaiagent.agent.agent.Agent`:

- :func:`traced_get_context` wraps ``memory.get_context(query)`` in a
  ``memory.read`` parent span and emits one ``memory.read.<block>`` child span
  per rendering block (counts, VectorBlock scores, bounded snippets).
- :func:`traced_add` wraps ``memory.add(message)`` in a ``memory.write`` parent
  span and emits one ``memory.write.<block>`` child span per block (action +
  detail).

Design notes:

- **No-op safe.** ``get_tracer()`` returns OTel's no-op tracer when tracing is
  off, so spans add negligible overhead and never change memory behaviour. No
  ``if tracer:`` guards needed.
- **Mirror KB exactly.** Attributes are set directly on the span and
  payload-bearing ones (``memory.query``, ``memory.snippets``, ``memory.detail``)
  are gated by ``trace_payloads_enabled()`` — we do *not* register them in
  ``FASTAIAGENT_ATTRIBUTES`` (neither does ``retrieval.*``).
- **No private reflection.** Per-block detail comes from each block's optional
  :meth:`MemoryBlock.last_render_report` / :meth:`last_write_report`; blocks that
  don't implement them are reported with safe defaults.
- **Contract preserved.** ``get_context`` still returns ``list[Message]`` and
  ``add`` still returns ``None`` — the helpers are transparent pass-throughs.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from fastaiagent.trace.span import trace_payloads_enabled

if TYPE_CHECKING:
    from fastaiagent.agent.memory import AgentMemory, ComposableMemory
    from fastaiagent.agent.memory_blocks import (
        BlockRenderReport,
        BlockWriteReport,
        MemoryBlock,
    )
    from fastaiagent.llm.message import Message

logger = logging.getLogger(__name__)


def _blocks(memory: Any) -> list[MemoryBlock]:
    """Return the memory's blocks, or ``[]`` for a plain ``AgentMemory``."""
    blocks = getattr(memory, "blocks", None)
    return list(blocks) if blocks else []


def _block_label(block: MemoryBlock) -> str:
    """A stable, span-name-safe label for a block."""
    return getattr(block, "name", None) or type(block).__name__


def _safe_render_report(block: MemoryBlock) -> BlockRenderReport | None:
    try:
        return block.last_render_report()
    except Exception:
        logger.debug("last_render_report failed for %r", block, exc_info=True)
        return None


def _safe_write_report(block: MemoryBlock) -> BlockWriteReport | None:
    try:
        return block.last_write_report()
    except Exception:
        logger.debug("last_write_report failed for %r", block, exc_info=True)
        return None


def traced_get_context(
    memory: AgentMemory | ComposableMemory,
    query: str,
) -> list[Message]:
    """Call ``memory.get_context(query=query)`` inside a ``memory.read`` span.

    Emits one ``memory.read.<block>`` child span per rendering block. Returns the
    unchanged ``list[Message]`` so the call site behaves identically to a bare
    ``memory.get_context`` call.
    """
    from fastaiagent.trace.otel import get_tracer

    tracer = get_tracer("fastaiagent.memory")
    blocks = _blocks(memory)
    start = time.monotonic()
    with tracer.start_as_current_span("memory.read") as span:
        messages = memory.get_context(query=query)

        span.set_attribute("fastaiagent.runner.type", "memory")
        span.set_attribute("memory.operation", "read")
        span.set_attribute("memory.block_count", len(blocks))
        span.set_attribute("memory.message_count", len(messages))
        if trace_payloads_enabled() and query:
            span.set_attribute("memory.query", query)

        for block in blocks:
            report = _safe_render_report(block)
            if report is None:
                # Custom/third-party block with no reporting interface — emit a
                # minimal child span so it's still visible, no private reflection.
                _emit_render_child(tracer, _block_label(block), type(block).__name__, None)
            else:
                _emit_render_child(
                    tracer, report.block_name or _block_label(block), report.block_type, report
                )

        span.set_attribute("memory.latency_ms", int((time.monotonic() - start) * 1000))
    return messages


def _emit_render_child(
    tracer: Any,
    label: str,
    block_type: str,
    report: BlockRenderReport | None,
) -> None:
    with tracer.start_as_current_span(f"memory.read.{label}") as child:
        child.set_attribute("fastaiagent.runner.type", "memory")
        child.set_attribute("memory.operation", "read")
        child.set_attribute("memory.block_name", label)
        child.set_attribute("memory.block_type", block_type)
        if report is None:
            child.set_attribute("memory.rendered_count", 0)
            return
        child.set_attribute("memory.rendered_count", report.rendered_count)
        if report.deduped_count is not None:
            child.set_attribute("memory.deduped_count", report.deduped_count)
        if report.scores:
            try:
                child.set_attribute("memory.scores", json.dumps(report.scores[:50]))
            except (TypeError, ValueError):
                logger.debug("Failed to serialize memory.scores", exc_info=True)
        if trace_payloads_enabled() and report.snippets:
            try:
                child.set_attribute("memory.snippets", json.dumps(report.snippets[:50]))
            except (TypeError, ValueError):
                logger.debug("Failed to serialize memory.snippets", exc_info=True)


def traced_add(
    memory: AgentMemory | ComposableMemory,
    message: Message,
) -> None:
    """Call ``memory.add(message)`` inside a ``memory.write`` span.

    Emits one ``memory.write.<block>`` child span per block describing what it
    did (``stored`` / ``summarized`` / ``extracted_facts`` / ``embedded`` /
    ``noop``).
    """
    from fastaiagent.trace.otel import get_tracer

    tracer = get_tracer("fastaiagent.memory")
    blocks = _blocks(memory)
    start = time.monotonic()
    with tracer.start_as_current_span("memory.write") as span:
        memory.add(message)

        span.set_attribute("fastaiagent.runner.type", "memory")
        span.set_attribute("memory.operation", "write")
        span.set_attribute("memory.messages_added", 1)
        span.set_attribute("memory.block_count", len(blocks))

        for block in blocks:
            report = _safe_write_report(block)
            label = (report.block_name if report else None) or _block_label(block)
            block_type = report.block_type if report else type(block).__name__
            action = report.action if report else "noop"
            detail = report.detail if report else None
            with tracer.start_as_current_span(f"memory.write.{label}") as child:
                child.set_attribute("fastaiagent.runner.type", "memory")
                child.set_attribute("memory.operation", "write")
                child.set_attribute("memory.block_name", label)
                child.set_attribute("memory.block_type", block_type)
                child.set_attribute("memory.action", action)
                if trace_payloads_enabled() and detail:
                    try:
                        child.set_attribute("memory.detail", json.dumps(detail))
                    except (TypeError, ValueError):
                        logger.debug("Failed to serialize memory.detail", exc_info=True)

        span.set_attribute("memory.latency_ms", int((time.monotonic() - start) * 1000))
