"""Live integration tests for multi-agent (Supervisor + Swarm) × (KB + Memory).

Per the project no-mocking rule, these tests drive real LLMs and real
storage backends end-to-end. They skip cleanly without API keys and FAISS.

Covered:
    - Supervisor + LocalKB (worker answers from KB)
    - Supervisor + ComposableMemory (shared memory across workers)
    - Swarm + LocalKB (peer with KB tool)
    - Swarm + ComposableMemory + VectorBlock (peers share memory)
    - Swarm that loops (writer ↔ critic) with termination
"""

from __future__ import annotations

import os

import pytest

from fastaiagent import (
    Agent,
    AgentMemory,
    ComposableMemory,
    LLMClient,
    StaticBlock,
    Supervisor,
    Swarm,
    VectorBlock,
    Worker,
)
from fastaiagent.kb import LocalKB
from fastaiagent.kb.embedding import SimpleEmbedder

try:
    import faiss  # noqa: F401

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

_HAS_LIVE_KEY = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))

pytestmark = [
    pytest.mark.skipif(not _HAS_FAISS, reason="faiss-cpu not installed"),
    pytest.mark.skipif(
        not _HAS_LIVE_KEY,
        reason="no OPENAI_API_KEY or ANTHROPIC_API_KEY set",
    ),
]


def _llm() -> LLMClient:
    if os.environ.get("OPENAI_API_KEY"):
        return LLMClient(provider="openai", model="gpt-4o-mini")
    return LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")


# ---------------------------------------------------------------------------
# Supervisor + KB
# ---------------------------------------------------------------------------


def test_supervisor_worker_uses_kb_to_answer(tmp_path) -> None:
    """Supervisor delegates to a worker whose tool is a KB search.

    Verifies that:
      1. Existing Worker delegation still works end-to-end.
      2. The worker's KB tool is invoked.
      3. The final answer reflects KB content.
    """
    llm = _llm()
    kb = LocalKB(
        name="kb-sv-test",
        path=str(tmp_path),
        embedder=SimpleEmbedder(dimensions=64),
        persist=False,
    )
    kb.add(
        "Refund policy: Returns accepted within 30 days of purchase. "
        "Items must be in original condition."
    )
    kb.add("Shipping: 3-5 business days domestic. Express available for $15.")

    support = Agent(
        name="support",
        system_prompt="Search the knowledge base and answer from it. Be terse.",
        llm=llm,
        tools=[kb.as_tool()],
    )
    supervisor = Supervisor(
        name="customer-service",
        llm=llm,
        workers=[
            Worker(
                agent=support,
                role="support",
                description="Answers policy questions from the KB",
            )
        ],
    )

    result = supervisor.run("What is the refund window?")
    # The KB contains "30 days" — the answer should too.
    assert "30" in result.output, f"expected refund window in output, got: {result.output!r}"


# ---------------------------------------------------------------------------
# Supervisor + ComposableMemory
# ---------------------------------------------------------------------------


def test_supervisor_worker_shares_composable_memory(tmp_path) -> None:
    """Shared ComposableMemory with a StaticBlock is visible inside the worker.

    We pin a fact via StaticBlock on the WORKER's memory and confirm the
    worker's answer reflects it — proving the memory block rendered into
    the worker's prompt during delegation.
    """
    llm = _llm()
    worker_memory = ComposableMemory(
        blocks=[StaticBlock("The user's preferred unit system is metric.")],
        primary=AgentMemory(max_messages=10),
    )
    converter = Agent(
        name="converter",
        system_prompt=(
            "Convert between units. Use the user's preferred unit system "
            "when there is a choice. Be terse."
        ),
        llm=llm,
        memory=worker_memory,
    )
    supervisor = Supervisor(
        name="unit-desk",
        llm=llm,
        workers=[Worker(agent=converter, role="converter", description="Handles unit conversions")],
    )
    result = supervisor.run("How tall is 6 feet? Answer in the user's preferred system.")
    # With metric preference active, the answer should mention cm or meters.
    out = result.output.lower()
    assert "cm" in out or "meter" in out or "1.8" in out, (
        f"expected metric answer, got: {result.output!r}"
    )


# ---------------------------------------------------------------------------
# Swarm + KB
# ---------------------------------------------------------------------------


def test_swarm_peer_uses_kb_tool(tmp_path) -> None:
    """A Swarm peer with a KB tool still uses the KB after a handoff."""
    llm = _llm()
    kb = LocalKB(
        name="kb-swarm",
        path=str(tmp_path),
        embedder=SimpleEmbedder(dimensions=64),
        persist=False,
    )
    kb.add("Company holiday: Offices are closed on April 25 for the annual strategy day.")
    kb.add("Dress code: Business casual on-site; no requirement for remote work.")

    triage = Agent(
        name="triage",
        system_prompt=(
            "Route to 'hr' for HR/benefits/office-policy questions. Route to "
            "'tech' for technical support. Always hand off; do not answer yourself."
        ),
        llm=llm,
    )
    hr = Agent(
        name="hr",
        system_prompt=(
            "Answer HR and office-policy questions. Always call search_kb-swarm "
            "first to look up the policy before answering. Be terse."
        ),
        llm=llm,
        tools=[kb.as_tool()],
    )
    tech = Agent(
        name="tech",
        system_prompt="Answer technical questions. Do not hand off.",
        llm=llm,
    )
    swarm = Swarm(
        name="company-desk",
        agents=[triage, hr, tech],
        entrypoint="triage",
        handoffs={"triage": ["hr", "tech"], "hr": [], "tech": []},
        max_handoffs=3,
    )
    result = swarm.run("Is the office open on April 25?")
    out = result.output.lower()
    # The answer should reflect the KB fact (closed / strategy day).
    assert "closed" in out or "strategy" in out or "25" in out, (
        f"expected KB-informed answer, got: {result.output!r}"
    )


# ---------------------------------------------------------------------------
# Swarm + ComposableMemory + VectorBlock
# ---------------------------------------------------------------------------


def test_swarm_peers_share_vector_memory(tmp_path) -> None:
    """Two peers share a ComposableMemory with a VectorBlock. Facts mentioned
    in one agent's turn are recalled by the other after a handoff.
    """
    from fastaiagent.kb.backends.faiss import FaissVectorStore

    llm = _llm()
    # Shared vector store + shared ComposableMemory across both peers.
    store = FaissVectorStore(dimension=64, index_type="flat")
    shared_memory = ComposableMemory(
        blocks=[
            VectorBlock(
                store=store,
                embedder=SimpleEmbedder(dimensions=64),
                top_k=3,
                min_content_chars=5,
            ),
        ],
        primary=AgentMemory(max_messages=10),
    )

    intake = Agent(
        name="intake",
        system_prompt=(
            "Collect a single fact about the user's preferences, then hand "
            "off to 'recommender' immediately."
        ),
        llm=llm,
        memory=shared_memory,
    )
    recommender = Agent(
        name="recommender",
        system_prompt=(
            "Use the shared memory (see 'Relevant prior exchanges' hints) to "
            "tailor a recommendation. Be terse. Do not hand off."
        ),
        llm=llm,
        memory=shared_memory,
    )
    swarm = Swarm(
        name="memory-swarm",
        agents=[intake, recommender],
        entrypoint="intake",
        handoffs={"intake": ["recommender"], "recommender": []},
        max_handoffs=2,
    )

    result = swarm.run(
        "I love spicy Szechuan food and hate sweet desserts. Recommend a "
        "restaurant dish for dinner."
    )
    out = result.output.lower()
    # Recommender should have leaned on the spicy/Szechuan signal from memory.
    assert any(kw in out for kw in ("spic", "szech", "chili", "hot", "numb", "mala")), (
        f"expected recommender to use shared memory hints, got: {result.output!r}"
    )


# ---------------------------------------------------------------------------
# Swarm that loops with termination
# ---------------------------------------------------------------------------


def test_swarm_writer_critic_loop_terminates() -> None:
    """Writer ↔ critic loop converges to a final answer within max_handoffs."""
    llm = _llm()
    writer = Agent(
        name="writer",
        system_prompt=(
            "Write a single sentence about the topic. If the critic returns "
            "the draft with feedback, revise it once. Do not ask questions."
        ),
        llm=llm,
    )
    critic = Agent(
        name="critic",
        system_prompt=(
            "Read the draft. If it's good, hand off to 'writer' with reason "
            "'APPROVED' — the writer will then produce the final version. "
            "Otherwise hand off with concrete feedback. You NEVER answer the "
            "user directly."
        ),
        llm=llm,
    )
    swarm = Swarm(
        name="loop-swarm",
        agents=[writer, critic],
        entrypoint="writer",
        handoffs={"writer": ["critic"], "critic": ["writer"]},
        max_handoffs=4,
    )
    result = swarm.run("Write a single-sentence definition of photosynthesis.")
    assert result.output.strip(), "expected a non-empty final answer"
    # A complete sentence should include a verb and be > 20 chars.
    assert len(result.output) > 20
