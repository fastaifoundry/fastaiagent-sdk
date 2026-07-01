"""Trace Learning Loop — extract durable facts from past traces.

Public surface::

    from fastaiagent.learn import (
        Fact,
        MemoryStore,
        ExtractionResult,
        extract_facts_from_trace,
        extract_and_store,
        run_extraction,
    )

The companion CLI is registered as ``fastaiagent learn`` (see
:mod:`fastaiagent.learn.cli`). Re-injection at agent runtime happens via
:class:`fastaiagent.agent.memory_blocks.PersistentFactBlock`.
"""

from __future__ import annotations

from fastaiagent.learn.extractor import (
    ExtractionResult,
    extract_and_store,
    extract_facts_from_trace,
    run_extraction,
)
from fastaiagent.learn.faststore import (
    FactStore,
    PostgresFactStore,
    RedisFactStore,
    SemanticFactStore,
    make_fact_store,
)
from fastaiagent.learn.store import Fact, MemoryStore, Scope

__all__ = [
    "ExtractionResult",
    "Fact",
    "FactStore",
    "MemoryStore",
    "PostgresFactStore",
    "RedisFactStore",
    "Scope",
    "SemanticFactStore",
    "extract_and_store",
    "extract_facts_from_trace",
    "make_fact_store",
    "run_extraction",
]
