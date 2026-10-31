"""Where a trace came from: production, or an experiment (1.89.0).

A replay rerun, a Playground call, a memory-learning pass, an eval case and an
AutoLLM candidate all produce ordinary-looking traces. Before 1.89.0 nothing
told them apart, so every one of them counted as production: in Analytics,
Overview and Agents, in eval curation and memory learning, and on the plane,
which received replays as real runs of the agent.

One mechanism stamps the origin:

* :func:`provenance_scope` is a context manager. Every span **started** inside
  it carries ``fastaiagent.source``; the scope's first span (its root)
  additionally carries the scope's ``attributes``. The innermost scope wins.
* Stamping happens in the SDK's own storage processor
  (:class:`~fastaiagent.trace.storage.LocalStorageProcessor.on_start`), so it
  covers every span the SDK writes, through ``run_sync`` and asyncio tasks
  alike (the scope is a ContextVar).
* A source in :data:`LOCAL_ONLY_SOURCES` also stamps
  ``fastaiagent.export.local_only``. Such rows are written already marked as
  sent (``synced=1``), so neither this SDK nor an older one sharing the store
  ever exports them, and ``add_exporter`` sinks drop them too. Opt back in with
  ``connect(export_experiments=True)``, ``FASTAIAGENT_EXPORT_EXPERIMENTS=1`` or
  a scope's ``export=True``.

Readers exclude experiments with :func:`non_production_filter` (SQL) or
:func:`is_non_production` (Python).

None of these attributes carries a payload; they are structural, like
``fastaiagent.meta.*``.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Collection, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

#: The span attribute that names a trace's origin. An open string: user code
#: may set its own values (``"batch-job"``), which count as production.
SOURCE_ATTR = "fastaiagent.source"
#: Present (True) on spans that must never leave the machine.
LOCAL_ONLY_ATTR = "fastaiagent.export.local_only"

#: Sources the SDK stamps. Every one is excluded from production views.
NON_PRODUCTION_SOURCES: frozenset[str] = frozenset(
    {"replay", "playground", "learn", "eval", "simulation", "optimize", "kb_search"}
)
#: Sources that are not exported by default. ``eval`` is deliberately absent:
#: the plane corroborates an exported eval verdict against its cases' traces.
LOCAL_ONLY_SOURCES: frozenset[str] = frozenset({"replay", "playground", "kb_search"})


@dataclass
class Provenance:
    """One active scope. Mutable only in :attr:`_scope_span_ids`."""

    source: str
    attributes: dict[str, Any] = field(default_factory=dict)
    export: bool | None = None
    trace_db: str | None = None
    tool_dispositions: dict[str, str] = field(default_factory=dict)
    suppress_registration: bool = False
    #: Span ids started inside this scope; a span whose parent is not among
    #: them is a scope root.
    _scope_span_ids: set[int] = field(default_factory=set)
    #: The trace of the first root span started in this scope (32-hex), so a
    #: caller that got no trace id back (an eval case running another
    #: framework's agent) can still name the run's trace (1.96.0).
    first_root_trace_id: str | None = None


_current: ContextVar[Provenance | None] = ContextVar("fastaiagent_provenance", default=None)


@contextlib.contextmanager
def provenance_scope(
    source: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    export: bool | None = None,
    trace_db: str | None = None,
    tool_dispositions: Mapping[str, str] | None = None,
    suppress_registration: bool = False,
) -> Iterator[Provenance]:
    """Stamp every span started in this block with ``fastaiagent.source=source``.

    Args:
        source: the origin, e.g. ``"replay"``. See :data:`NON_PRODUCTION_SOURCES`.
        attributes: set on the scope's root span only (e.g. the replay's
            ``fastaiagent.replay.*`` keys). Never put a payload here.
        export: ``True``/``False`` overrides whether a local-only source is
            exported; ``None`` defers to :func:`experiments_exported`.
        trace_db: write this scope's spans to another local store (replay's
            split-DB sink).
        tool_dispositions: ``{tool_name: disposition}``; stamped on that tool's
            spans as ``fastaiagent.replay.tool_disposition``.
        suppress_registration: no agent auto-registration or push to the plane
            while the scope is active (a rerun must not overwrite the
            definition of the agent it replays).
    """
    prov = Provenance(
        source=source,
        attributes=dict(attributes or {}),
        export=export,
        trace_db=trace_db,
        tool_dispositions=dict(tool_dispositions or {}),
        suppress_registration=suppress_registration,
    )
    token = _current.set(prov)
    try:
        yield prov
    finally:
        _current.reset(token)


def current_provenance() -> Provenance | None:
    """The innermost active scope, or ``None``."""
    return _current.get()


def registration_suppressed() -> bool:
    """True inside a scope that forbids agent auto-registration."""
    prov = _current.get()
    return bool(prov and prov.suppress_registration)


def _export_experiments_env() -> bool:
    """Resolver for ``FASTAIAGENT_EXPORT_EXPERIMENTS`` (registered in ``ENV_FLAGS``)."""
    from fastaiagent._internal.env import env_flag

    return env_flag("FASTAIAGENT_EXPORT_EXPERIMENTS", default=False, on_unparsed=False)


def experiments_exported(prov: Provenance | None = None) -> bool:
    """Whether a local-only source may be exported.

    The scope's ``export``, then ``connect(export_experiments=...)``, then
    ``FASTAIAGENT_EXPORT_EXPERIMENTS``, then ``False``.
    """
    if prov is not None and prov.export is not None:
        return prov.export
    try:
        from fastaiagent.client import _connection

        configured = getattr(_connection, "export_experiments", None)
        if configured is not None:
            return bool(configured)
    except Exception:  # pragma: no cover — the client module always imports
        pass
    return _export_experiments_env()


def stamp_on_start(span: Any) -> None:
    """Stamp provenance on a span as it starts. Called by the storage processor.

    Never raises: provenance is bookkeeping and must not break a run.
    """
    prov = _current.get()
    if prov is None:
        return
    try:
        ctx = span.get_span_context()
        parent = getattr(span, "parent", None)
        is_root = parent is None or parent.span_id not in prov._scope_span_ids
        prov._scope_span_ids.add(ctx.span_id)
        if is_root and prov.first_root_trace_id is None:
            prov.first_root_trace_id = format(ctx.trace_id, "032x")

        span.set_attribute(SOURCE_ATTR, prov.source)
        if prov.source in LOCAL_ONLY_SOURCES and not experiments_exported(prov):
            span.set_attribute(LOCAL_ONLY_ATTR, True)
        if is_root:
            for key, value in prov.attributes.items():
                if value is None:
                    continue
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                span.set_attribute(key, value)
        name = getattr(span, "name", "") or ""
        if name.startswith("tool.") and prov.tool_dispositions:
            disposition = prov.tool_dispositions.get(name[len("tool.") :])
            if disposition:
                span.set_attribute("fastaiagent.replay.tool_disposition", disposition)
    except Exception:  # pragma: no cover — bookkeeping never breaks the caller
        import logging

        logging.getLogger(__name__).debug("Could not stamp provenance", exc_info=True)


def is_non_production(attrs: Mapping[str, Any], *, include: bool | Collection[str] = False) -> bool:
    """True when one span's attributes mark it as an experiment.

    ``include=True`` treats nothing as an experiment; a collection of sources
    keeps those sources (e.g. ``{"eval"}``).
    """
    if include is True:
        return False
    source = attrs.get(SOURCE_ATTR)
    if source not in NON_PRODUCTION_SOURCES:
        return False
    return not (include and source in include)


def non_production_filter(
    *,
    column: str = "trace_id",
    include: bool | Collection[str] = False,
    mode: Literal["exclude", "only"] = "exclude",
) -> tuple[str, tuple[Any, ...]]:
    """SQL that excludes (or keeps only) traces with a non-production span.

    Returns ``("AND <column> NOT IN (...)", params)``, ready to append to a
    ``WHERE`` clause, in the shape of ``ui.deps.project_filter``. Trace-level:
    a trace is an experiment when **any** of its spans carries a
    non-production source, so a playground trace stamped on its root only (as
    before 1.89.0) and a rerun wrapped in a user's own span both count.

    ``include=True`` returns ``("", ())``. A collection of sources removes
    those sources from the exclusion. ``mode="only"`` inverts it.
    """
    if include is True:
        return "", ()
    keep = set(include) if include else set()
    sources = sorted(NON_PRODUCTION_SOURCES - keep)
    if not sources:
        return "", ()
    placeholders = ",".join("?" * len(sources))
    sub = (
        "SELECT _np.trace_id FROM spans AS _np "
        f"WHERE _np.attributes LIKE '%{SOURCE_ATTR}%' "
        "AND CASE WHEN json_valid(_np.attributes) "
        f"THEN json_extract(_np.attributes, '$.\"{SOURCE_ATTR}\"') END IN ({placeholders})"
    )
    op = "NOT IN" if mode == "exclude" else "IN"
    return f"AND {column} {op} ({sub})", tuple(sources)


__all__ = [
    "LOCAL_ONLY_ATTR",
    "LOCAL_ONLY_SOURCES",
    "NON_PRODUCTION_SOURCES",
    "SOURCE_ATTR",
    "Provenance",
    "current_provenance",
    "experiments_exported",
    "is_non_production",
    "non_production_filter",
    "provenance_scope",
    "registration_suppressed",
    "stamp_on_start",
]
