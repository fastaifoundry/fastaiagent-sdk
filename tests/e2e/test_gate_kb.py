"""End-to-end quality gate — LocalKB (knowledge base / RAG).

Exercises the keyword and (where available) hybrid search paths plus
SQLite persistence. Each gate sub-test is self-contained with an
isolated temp dir, so running the gate doesn't pollute the developer's
local ``.fastaiagent/kb/`` directory.

Keyword search is always exercised because it has no heavyweight
dependencies. Hybrid search (FAISS + BM25 + embedder) is exercised
only when ``fastembed`` is importable, so this gate stays green in
environments that installed the SDK without the ``[kb]`` extra.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


_FASTEMBED_AVAILABLE = importlib.util.find_spec("fastembed") is not None


_SEED_DOCS = [
    (
        "Refund Policy v4.2: Customers may request a full refund within 30 days "
        "of purchase. After 30 days, store credit is offered in lieu of cash."
    ),
    (
        "Shipping Policy: Standard shipping takes 3-5 business days. Express "
        "shipping is 1-2 business days. International orders ship via DHL."
    ),
    (
        "Warranty: All electronics carry a one-year limited warranty covering "
        "manufacturing defects. Accidental damage is not covered."
    ),
]


class TestLocalKBKeywordGate:
    """Keyword-only LocalKB — no embedder dependency required."""

    def test_01_add_and_search_keyword(self, tmp_path: Path) -> None:
        require_env()
        from fastaiagent import LocalKB

        kb = LocalKB(
            name="gate-keyword",
            path=str(tmp_path),
            search_type="keyword",
            persist=True,
        )
        try:
            total = 0
            for doc in _SEED_DOCS:
                total += kb.add(doc)
            assert total >= len(_SEED_DOCS), (
                f"expected >= {len(_SEED_DOCS)} chunks, got {total}"
            )

            results = kb.search("refund within 30 days", top_k=3)
            assert results, "keyword search returned zero results"
            assert any(
                "refund" in r.chunk.content.lower() for r in results
            ), "top results do not mention 'refund'"
        finally:
            kb.close()

    def test_02_persistence_round_trip(self, tmp_path: Path) -> None:
        require_env()
        from fastaiagent import LocalKB

        kb1 = LocalKB(
            name="gate-persistence",
            path=str(tmp_path),
            search_type="keyword",
            persist=True,
        )
        try:
            kb1.add(_SEED_DOCS[0])
            kb1.add(_SEED_DOCS[1])
        finally:
            kb1.close()

        # Re-open against the same path — chunks should auto-load from SQLite.
        kb2 = LocalKB(
            name="gate-persistence",
            path=str(tmp_path),
            search_type="keyword",
            persist=True,
        )
        try:
            results = kb2.search("shipping international", top_k=3)
            assert results, (
                "re-opened KB returned zero results — persistence is broken"
            )
            assert any(
                "dhl" in r.chunk.content.lower() or "shipping" in r.chunk.content.lower()
                for r in results
            ), "persistence round-trip lost the shipping document"
        finally:
            kb2.close()


@pytest.mark.skipif(
    not _FASTEMBED_AVAILABLE,
    reason="fastembed not installed — vector/hybrid search path unavailable",
)
class TestLocalKBHybridGate:
    """Hybrid (FAISS + BM25 + embedder) path — only when fastembed is installed."""

    def test_01_hybrid_search_retrieves_semantic_match(
        self, tmp_path: Path
    ) -> None:
        require_env()
        from fastaiagent import LocalKB

        kb = LocalKB(
            name="gate-hybrid",
            path=str(tmp_path),
            search_type="hybrid",
            index_type="flat",
            persist=True,
        )
        try:
            for doc in _SEED_DOCS:
                kb.add(doc)

            # Query uses different words than the source ("money back" vs "refund")
            # so we exercise semantic (vector) matching, not just keyword.
            results = kb.search("get my money back after a month", top_k=3)
            assert results, "hybrid search returned zero results"
            # Top hit should be the refund policy even without lexical overlap.
            top_content = results[0].chunk.content.lower()
            assert (
                "refund" in top_content or "30 days" in top_content
            ), (
                f"hybrid search missed the semantic match — "
                f"top hit was: {top_content!r}"
            )
        finally:
            kb.close()
