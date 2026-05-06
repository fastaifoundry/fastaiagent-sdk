"""
Memory wiring — closes the trace-learning loop.

Returns an :class:`AgentMemory` (actually a :class:`ComposableMemory`) that
injects facts learned from past traces by ``fastaiagent learn``. The block
is read-only at runtime; the offline CLI is what writes new facts.

The agent scope (``scope="agent"`` + ``scope_id="deep-research"``) makes
this template's learned facts live alongside its own traces, not mixed
with other templates' memories.

Set ``DEEP_RESEARCH_DISABLE_LEARNED_MEMORY=1`` to bypass the block — for
A/B comparing the agent with and without learned facts.
"""

from __future__ import annotations

import os
from typing import Any

import fastaiagent as fa

SCOPE = "agent"
SCOPE_ID = "deep-research"


def build_memory() -> Any | None:
    """Return a :class:`ComposableMemory` wired with the persistent fact block.

    Returns ``None`` if learning is explicitly disabled or the learn module
    is unavailable for any reason — the pipeline still works without it.
    """
    if os.environ.get("DEEP_RESEARCH_DISABLE_LEARNED_MEMORY", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return None

    try:
        block = fa.PersistentFactBlock(
            scope=SCOPE,
            scope_id=SCOPE_ID,
            project_id=os.environ.get("FASTAIAGENT_PROJECT_ID", ""),
            max_facts=int(os.environ.get("DEEP_RESEARCH_MAX_LEARNED_FACTS", "30")),
        )
    except Exception:
        # If the table doesn't exist yet (older SDK) or anything else
        # fails, silently fall back to no learned memory rather than
        # crash the pipeline.
        return None

    return fa.ComposableMemory(primary=fa.AgentMemory(), blocks=[block])
