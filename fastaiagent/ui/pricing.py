"""Re-export of the model pricing table, which now lives in ``_internal``.

The table moved to :mod:`fastaiagent._internal.pricing` in 1.67.0 because it is
on the LLM hot path: ``LLMClient`` prices every completion so
``AgentResult.cost`` and the ``fastaiagent.cost.total_usd`` span attribute are
real numbers rather than a permanent ``0.0``. Importing a ``ui``-namespaced
module from ``llm/client.py`` would have dragged the UI namespace into core.

Nothing about the table changed, and this module keeps the whole public
surface, so every existing ``from fastaiagent.ui.pricing import ...`` — the UI
routes, the trace exporter, the langchain/crewai/pydanticai integrations —
keeps working unchanged. Prefer the ``_internal`` path in new core code and
this one in UI code.
"""

from __future__ import annotations

from fastaiagent._internal.pricing import (
    _PRICING,
    _active_overrides,
    _match,
    _parse_rate,
    _Rate,
    compute_cost_usd,
    reload_rate_overrides,
    set_rate_overrides,
    usage_cost,
)

__all__ = [
    "_PRICING",
    "_Rate",
    "_active_overrides",
    "_match",
    "_parse_rate",
    "compute_cost_usd",
    "reload_rate_overrides",
    "set_rate_overrides",
    "usage_cost",
]
