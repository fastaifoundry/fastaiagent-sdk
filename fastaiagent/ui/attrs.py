"""Tolerant attribute lookup for span attribute dicts.

The live SDK emits unprefixed attribute names (``agent.name``,
``agent.tokens_used``, ``agent.latency_ms``, etc.). We also support two
namespaced variants:

* ``fastaiagent.*`` — the canonical prefix going forward for anything the
  SDK adds on top of the standard OTel ``gen_ai.*`` namespace.
* ``fastai.*`` — legacy; never emit it, but accept it on read so old
  traces don't suddenly go blank after upgrading.

``fastai.*`` is reserved by fast.ai (a different company); we never write
new attributes under that prefix. See
``feedback_no_fastai_namespace.md``.

Use :func:`attr` to read any custom field:

    attr(span_attrs, "agent.name")
    # tries fastaiagent.agent.name → agent.name → fastai.agent.name
"""

from __future__ import annotations

from typing import Any


def attr(attrs: dict[str, Any] | None, key: str) -> Any:
    """Read ``key`` from a span attributes dict, tolerating prefix variants.

    Search order:
    1. ``fastaiagent.<key>`` (canonical namespaced).
    2. ``<key>`` (unprefixed — the live SDK's current convention).
    3. ``fastai.<key>`` (legacy).

    Returns the first non-``None`` value found, or ``None`` if none match.
    """
    if not attrs:
        return None
    for candidate in (f"fastaiagent.{key}", key, f"fastai.{key}"):
        value = attrs.get(candidate)
        if value is not None:
            return value
    return None


def trace_cost_usd(attrs: dict[str, Any] | None) -> float | None:
    """Best-effort cost lookup across every attribute variant.

    Checks ``agent.cost_usd`` first (what the SDK will emit once cost
    tracking lands), then falls back to the historical ``cost.total_usd``
    key under either prefix.
    """
    for key in ("agent.cost_usd", "cost.total_usd"):
        value = attr(attrs, key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None
