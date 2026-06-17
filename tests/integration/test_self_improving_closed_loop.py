"""Closed-loop integration test for the Trace Learning Loop (PR B).

Walks the full self-improving story end-to-end:

  1. Run a small Agent to seed traces in an isolated local.db.
  2. Run the offline learn extractor — facts persist to learned_memory.
  3. Construct a NEW Agent with PersistentFactBlock — its system prompt
     now carries the learned facts.
  4. Verify the trace from step 3 contains the injected SystemMessage.

Gated on OPENAI_API_KEY. Uses gpt-4o-mini throughout to keep cost low.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for closed-loop e2e",
)


def test_seed_then_learn_then_replay(isolated_local_db, tmp_path: Path) -> None:
    # ``isolated_local_db`` points FASTAIAGENT_LOCAL_DB at ``tmp_path/local.db`` AND
    # clears the cached config (setup + teardown), so MemoryStore() reads the temp
    # store regardless of whether an earlier test already cached get_config() — the
    # facts no longer accumulate across runs in the real local.db.
    import fastaiagent as fa
    from fastaiagent.learn import MemoryStore, run_extraction

    SCOPE_ID = "closed-loop-test"
    llm = fa.LLMClient(provider="openai", model="gpt-4o-mini")

    # ── Phase 1 — seed
    seed_agent = fa.Agent(
        name=SCOPE_ID,
        system_prompt=(
            "You are a strict code-style assistant. "
            "When asked, always remind the user that this codebase uses "
            "snake_case for Python variables and PEP 8 for formatting."
        ),
        llm=llm,
    )

    async def seed():
        for prompt in [
            "What's our naming convention again?",
            "Remind me how we format Python.",
        ]:
            await seed_agent.arun(prompt)

    asyncio.run(seed())

    # ── Phase 2 — extract
    store = MemoryStore()
    results = run_extraction(
        llm=llm,
        store=store,
        scope="agent",
        scope_id=SCOPE_ID,
        last_hours=1,
        max_facts_per_trace=10,
    )
    written = sum(len(r.written_ids) for r in results)
    # Real LLM may return 0 candidates legitimately; just assert the
    # pipeline ran cleanly and the store table is queryable.
    assert all(r.error is None for r in results), (
        f"extraction errored: {[r.error for r in results if r.error]}"
    )
    active = store.list_active(scope="agent", scope_id=SCOPE_ID)
    assert len(active) == written

    # If the model produced nothing durable, skip the prompt-injection
    # assertion gracefully — that's a quality issue with the prompt, not
    # a bug in the loop. The loop itself is what we're testing.
    if written == 0:
        pytest.skip("LLM returned 0 candidate facts — extraction worked, nothing to inject")

    # ── Phase 3 — replay with PersistentFactBlock
    memory = fa.ComposableMemory(
        primary=fa.AgentMemory(),
        blocks=[
            fa.PersistentFactBlock(scope="agent", scope_id=SCOPE_ID, max_facts=20),
        ],
    )
    replay_agent = fa.Agent(
        name=f"{SCOPE_ID}-replay",
        system_prompt="You are a Python style assistant.",
        llm=llm,
        memory=memory,
    )
    asyncio.run(replay_agent.arun("Quick reminder: how do we name variables?"))

    # ── Verify the replay trace carried the learned facts
    from fastaiagent._internal.storage import SQLiteHelper

    db = SQLiteHelper(str(tmp_path / "local.db"))
    try:
        rows = db.fetchall(
            "SELECT name, attributes FROM spans WHERE name LIKE 'llm.%'"
        )
    finally:
        db.close()

    # At least one LLM span should have the learned-facts SystemMessage in
    # the gen_ai.request.messages payload. We don't assert on the exact
    # text the model produced — just that the prompt injection happened.
    found_injection = False
    for row in rows:
        attrs = json.loads(row["attributes"]) if row["attributes"] else {}
        prompt_payload = attrs.get("gen_ai.request.messages", "")
        if "Learned facts" in str(prompt_payload):
            found_injection = True
            break
    assert found_injection, (
        "PersistentFactBlock did not inject its SystemMessage into any LLM "
        "request — the closed loop is broken somewhere"
    )
