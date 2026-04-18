"""Integration tests for ``PlatformKB`` against a live FastAIAgent platform.

Gated on ``FA_TEST_API_KEY`` (or ``FASTAIAGENT_API_KEY``). Skipped if unset so
CI can opt-in. No mocks — talks to the real platform, real DB, real retriever.

Required environment variables:
    FA_TEST_API_KEY      (or FASTAIAGENT_API_KEY) — API key with ``kb:read``
    FA_TEST_KB_ID        — UUID of a seeded KB accessible to the key
    FA_TEST_BASE_URL     (or FASTAIAGENT_TARGET) — platform URL, default localhost:8001

Run:
    pytest tests/integration/test_platform_kb.py -v
"""

from __future__ import annotations

import asyncio
import os

import pytest

import fastaiagent as fa
from fastaiagent._internal.errors import PlatformNotFoundError
from fastaiagent.kb.search import SearchResult


def _env(name: str, *fallbacks: str, default: str | None = None) -> str | None:
    for key in (name, *fallbacks):
        val = os.environ.get(key)
        if val:
            return val
    return default


API_KEY = _env("FA_TEST_API_KEY", "FASTAIAGENT_API_KEY")
KB_ID = _env("FA_TEST_KB_ID")
BASE_URL = _env("FA_TEST_BASE_URL", "FASTAIAGENT_TARGET", default="http://localhost:8001")

pytestmark = pytest.mark.skipif(
    not (API_KEY and KB_ID),
    reason="requires live platform; set FA_TEST_API_KEY and FA_TEST_KB_ID",
)


@pytest.fixture(scope="module", autouse=True)
def connected():
    fa.connect(api_key=API_KEY or "", target=BASE_URL or "")
    yield
    fa.disconnect()


def test_platform_kb_search_returns_results():
    kb = fa.PlatformKB(kb_id=KB_ID or "")
    results = kb.search("refund policy", top_k=3)

    assert len(results) > 0
    assert len(results) <= 3
    for r in results:
        assert isinstance(r, SearchResult)
        assert r.chunk.id
        assert r.chunk.content
        assert isinstance(r.score, float)


def test_platform_kb_search_async():
    kb = fa.PlatformKB(kb_id=KB_ID or "")
    results = asyncio.run(kb.asearch("refund policy", top_k=3))
    assert len(results) > 0
    assert len(results) <= 3
    assert all(isinstance(r, SearchResult) for r in results)


def test_platform_kb_top_k_honored():
    kb = fa.PlatformKB(kb_id=KB_ID or "")
    one = kb.search("policy", top_k=1)
    many = kb.search("policy", top_k=5)
    assert len(one) <= 1
    assert len(many) <= 5
    # Every non-trivial KB should have >=1 for a generic query.
    assert len(many) >= len(one)


def test_platform_kb_invalid_id_raises():
    kb = fa.PlatformKB(kb_id="00000000-0000-0000-0000-000000000000")
    with pytest.raises(PlatformNotFoundError):
        kb.search("anything")


def test_platform_kb_empty_kb_id_rejected():
    with pytest.raises(ValueError):
        fa.PlatformKB(kb_id="")


def test_platform_kb_swappable_with_local_kb():
    """Agent wires PlatformKB via ``as_tool()`` — identical to LocalKB."""
    kb = fa.PlatformKB(kb_id=KB_ID or "")
    tool = kb.as_tool()
    assert tool.name.startswith("search_")

    # Duck-type parity: .search() returns list[SearchResult]
    results = kb.search("test", top_k=1)
    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)


def test_platform_kb_metadata_passthrough():
    """Platform enriches chunks with document_name / score metadata."""
    kb = fa.PlatformKB(kb_id=KB_ID or "")
    results = kb.search("refund", top_k=2)
    assert results
    first = results[0]
    # Platform always ships at least these fields through the response.
    assert "document_name" in first.chunk.metadata
    assert "source_type" in first.chunk.metadata


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for agent end-to-end",
)
def test_agent_uses_platform_kb_end_to_end():
    """Agent with ``PlatformKB.as_tool()`` answers using retrieved content."""
    kb = fa.PlatformKB(kb_id=KB_ID or "")
    agent = fa.Agent(
        name="policy-bot",
        system_prompt=(
            "Answer from the provided knowledge base only. "
            "Use the search tool before answering."
        ),
        llm=fa.LLMClient(provider="openai", model="gpt-4o-mini"),
        tools=[kb.as_tool()],
    )
    result = agent.run("What's the refund policy?")
    assert result.output
    assert len(result.tool_calls) >= 1
    assert any(tc["tool_name"].startswith("search_") for tc in result.tool_calls)
