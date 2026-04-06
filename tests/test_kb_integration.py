"""Integration tests for LocalKB with real embeddings (FastEmbed), real data,
persistence, hybrid search, CRUD, and directory ingestion.

These tests use actual FastEmbed embeddings (BAAI/bge-small-en-v1.5) and
realistic document content — no mocking.

Requirements: pip install fastaiagent[kb]
"""

from __future__ import annotations

import textwrap

import pytest

from fastaiagent.kb import LocalKB

try:
    from fastaiagent.kb.embedding import FastEmbedEmbedder

    _HAS_FASTEMBED = True
except ImportError:
    _HAS_FASTEMBED = False

pytestmark = pytest.mark.skipif(not _HAS_FASTEMBED, reason="fastembed not installed")


@pytest.fixture(scope="module")
def embedder():
    """Shared FastEmbed embedder (model loads once for all tests)."""
    return FastEmbedEmbedder()


# ---------------------------------------------------------------------------
# Realistic document corpus
# ---------------------------------------------------------------------------

SUPPORT_DOCS = {
    "refund_policy": textwrap.dedent("""\
        Refund Policy

        All purchases are eligible for a full refund within 30 days of the original
        purchase date. Items must be returned in their original packaging and in unused
        condition. Digital products, gift cards, and subscription services are non-refundable.

        To initiate a return, navigate to My Orders > Select the item > Request Return.
        Refunds are processed within 5-7 business days to your original payment method.
        Shipping costs for returns are the customer's responsibility unless the item
        arrived damaged or defective.
    """),
    "shipping_guide": textwrap.dedent("""\
        Shipping Guide

        Domestic orders ship within 1-2 business days and arrive in 3-5 business days
        via standard shipping. Express shipping (1-2 day delivery) is available for an
        additional $9.99. Free shipping on orders over $75.

        International shipping takes 7-14 business days. Customs duties and import taxes
        are the responsibility of the buyer. We ship to over 50 countries worldwide.

        Order tracking is available at track.example.com. You will receive a tracking
        number via email once your order has shipped.
    """),
    "error_codes": textwrap.dedent("""\
        Error Code Reference

        ERR-4012: Payment gateway timeout. The payment processor did not respond within
        the expected timeframe. Wait 30 seconds and retry. If the issue persists, check
        your card details or try a different payment method.

        ERR-5001: Authentication failure. Your session has expired or your credentials
        are invalid. Log out, clear your browser cache, and log in again.

        ERR-3007: Inventory sync error. The item you're trying to purchase is temporarily
        out of stock. Check back in 24 hours or enable stock notifications.

        ERR-6100: Rate limit exceeded. You've made too many requests in a short period.
        Wait 60 seconds before retrying. Contact support if this persists.
    """),
    "faq": textwrap.dedent("""\
        Frequently Asked Questions

        Q: Can I change my shipping address after placing an order?
        A: Yes, within 1 hour of placing the order. Go to My Orders > Edit Address.
        After 1 hour, contact support for assistance.

        Q: Do you offer student discounts?
        A: Yes! Verify your student status at student.example.com for 15% off.

        Q: How do I cancel my subscription?
        A: Go to Account Settings > Subscriptions > Cancel. Cancellations take effect
        at the end of your current billing period. No partial refunds.

        Q: What payment methods do you accept?
        A: Visa, Mastercard, American Express, PayPal, Apple Pay, and Google Pay.
    """),
}


# ---------------------------------------------------------------------------
# Hybrid search — semantic + keyword
# ---------------------------------------------------------------------------


class TestHybridSearchIntegration:
    """Test hybrid search with real FastEmbed embeddings."""

    def test_semantic_query(self, temp_dir, embedder):
        """Natural language query matches semantically."""
        kb = LocalKB(name="test", path=str(temp_dir), embedder=embedder, persist=False)
        for text in SUPPORT_DOCS.values():
            kb.add(text)

        results = kb.search("how do I get my money back", top_k=3)
        assert len(results) >= 1
        # Top result should be from refund policy
        top_content = results[0].chunk.content.lower()
        assert "refund" in top_content or "return" in top_content

    def test_exact_code_query(self, temp_dir, embedder):
        """Exact error code matched via BM25 in hybrid mode."""
        kb = LocalKB(name="test", path=str(temp_dir), embedder=embedder, persist=False)
        for text in SUPPORT_DOCS.values():
            kb.add(text)

        results = kb.search("ERR-4012", top_k=3)
        assert len(results) >= 1
        top_content = results[0].chunk.content
        assert "ERR-4012" in top_content

    def test_mixed_query(self, temp_dir, embedder):
        """Query mixing natural language + error code."""
        kb = LocalKB(name="test", path=str(temp_dir), embedder=embedder, persist=False)
        for text in SUPPORT_DOCS.values():
            kb.add(text)

        results = kb.search("ERR-5001 I can't log in", top_k=3)
        assert len(results) >= 1
        top_content = results[0].chunk.content
        assert "ERR-5001" in top_content or "authentication" in top_content.lower()

    def test_question_answer_style(self, temp_dir, embedder):
        """FAQ-style question finds the right answer."""
        kb = LocalKB(name="test", path=str(temp_dir), embedder=embedder, persist=False)
        for text in SUPPORT_DOCS.values():
            kb.add(text)

        results = kb.search("do you have student discounts?", top_k=3)
        assert len(results) >= 1
        top_content = results[0].chunk.content.lower()
        assert "student" in top_content


# ---------------------------------------------------------------------------
# Vector-only search
# ---------------------------------------------------------------------------


class TestVectorOnlyIntegration:
    def test_semantic_search(self, temp_dir, embedder):
        kb = LocalKB(
            name="test", path=str(temp_dir), embedder=embedder,
            search_type="vector", persist=False,
        )
        for text in SUPPORT_DOCS.values():
            kb.add(text)

        results = kb.search("international delivery options", top_k=2)
        assert len(results) >= 1
        top = results[0].chunk.content.lower()
        assert "shipping" in top or "international" in top


# ---------------------------------------------------------------------------
# Keyword-only search
# ---------------------------------------------------------------------------


class TestKeywordOnlyIntegration:
    def test_exact_match_no_embedder(self, temp_dir):
        """Keyword search works without any embedder."""
        kb = LocalKB(
            name="test", path=str(temp_dir),
            search_type="keyword", persist=False,
        )
        for text in SUPPORT_DOCS.values():
            kb.add(text)

        results = kb.search("ERR-6100", top_k=1)
        assert len(results) == 1
        assert "ERR-6100" in results[0].chunk.content

    def test_no_semantic_understanding(self, temp_dir):
        """Keyword search doesn't find semantic matches without shared terms."""
        kb = LocalKB(
            name="test", path=str(temp_dir),
            search_type="keyword", persist=False,
        )
        kb.add("The refund policy covers returns within 30 days.")
        results = kb.search("get my money back", top_k=5)
        # BM25 shouldn't match — no shared keywords (money/back not in doc)
        for r in results:
            assert r.score < 1.0  # Low score or no match


# ---------------------------------------------------------------------------
# Persistence with real embeddings
# ---------------------------------------------------------------------------


class TestPersistenceIntegration:
    def test_persist_and_reload(self, temp_dir, embedder):
        """Data survives across KB instances with real embeddings."""
        kb1 = LocalKB(name="persist-test", path=str(temp_dir), embedder=embedder)
        kb1.add(SUPPORT_DOCS["refund_policy"])
        kb1.add(SUPPORT_DOCS["shipping_guide"])
        count = kb1.status()["chunk_count"]
        kb1.close()

        # Reload — should NOT re-embed
        kb2 = LocalKB(name="persist-test", path=str(temp_dir), embedder=embedder)
        assert kb2.status()["chunk_count"] == count

        # Search should still work
        results = kb2.search("how to return an item", top_k=2)
        assert len(results) >= 1
        assert "return" in results[0].chunk.content.lower() or "refund" in results[0].chunk.content.lower()
        kb2.close()

    def test_delete_persists(self, temp_dir, embedder):
        """Delete survives reload."""
        kb1 = LocalKB(name="del-test", path=str(temp_dir), embedder=embedder)
        kb1.add("Content A to keep")
        kb1.add("Content B to delete")
        chunk_id = kb1._chunks[1].id
        kb1.delete(chunk_id)
        kb1.close()

        kb2 = LocalKB(name="del-test", path=str(temp_dir), embedder=embedder)
        assert kb2.status()["chunk_count"] == 1
        assert kb2._chunks[0].content.startswith("Content A")
        kb2.close()

    def test_update_persists(self, temp_dir, embedder):
        """Update survives reload and search still works."""
        kb1 = LocalKB(name="upd-test", path=str(temp_dir), embedder=embedder)
        kb1.add("Original shipping policy text about delivery times.")
        chunk_id = kb1._chunks[0].id
        kb1.update(chunk_id, "Updated refund policy text about return procedures.")
        kb1.close()

        kb2 = LocalKB(name="upd-test", path=str(temp_dir), embedder=embedder)
        assert "Updated refund" in kb2._chunks[0].content
        results = kb2.search("return procedures", top_k=1)
        assert len(results) >= 1
        kb2.close()


# ---------------------------------------------------------------------------
# CRUD with real embeddings
# ---------------------------------------------------------------------------


class TestCRUDIntegration:
    def test_delete_by_source(self, temp_dir, embedder):
        """Delete by source removes all chunks from a file."""
        doc_path = temp_dir / "test_doc.txt"
        doc_path.write_text(SUPPORT_DOCS["error_codes"])

        kb = LocalKB(name="crud-test", path=str(temp_dir), embedder=embedder, persist=False)
        kb.add(str(doc_path))
        kb.add("Some other content not from the file.")

        initial = kb.status()["chunk_count"]
        deleted = kb.delete_by_source(str(doc_path))
        assert deleted > 0
        assert kb.status()["chunk_count"] == initial - deleted

        # Search should not find error codes anymore
        results = kb.search("ERR-4012", top_k=5)
        for r in results:
            assert "ERR-4012" not in r.chunk.content

    def test_clear_and_rebuild(self, temp_dir, embedder):
        """Clear wipes everything, new data works after."""
        kb = LocalKB(name="clear-test", path=str(temp_dir), embedder=embedder, persist=False)
        kb.add(SUPPORT_DOCS["refund_policy"])
        assert kb.status()["chunk_count"] > 0

        kb.clear()
        assert kb.status()["chunk_count"] == 0

        kb.add(SUPPORT_DOCS["shipping_guide"])
        results = kb.search("express shipping", top_k=1)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Directory ingestion
# ---------------------------------------------------------------------------


class TestDirectoryIngestionIntegration:
    def test_ingest_directory_of_docs(self, temp_dir, embedder):
        """Ingest a directory with multiple text files."""
        docs_dir = temp_dir / "support_docs"
        docs_dir.mkdir()
        (docs_dir / "refund.txt").write_text(SUPPORT_DOCS["refund_policy"])
        (docs_dir / "shipping.md").write_text(SUPPORT_DOCS["shipping_guide"])
        (docs_dir / "errors.txt").write_text(SUPPORT_DOCS["error_codes"])
        (docs_dir / "faq.md").write_text(SUPPORT_DOCS["faq"])
        # Unsupported file — should be ignored
        (docs_dir / "config.json").write_text('{"version": 1}')

        kb = LocalKB(name="dir-test", path=str(temp_dir), embedder=embedder, persist=False)
        count = kb.add(str(docs_dir))
        assert count > 4  # Multiple chunks across 4 files

        # Search across all ingested docs
        results = kb.search("how to cancel subscription", top_k=2)
        assert len(results) >= 1
        top = results[0].chunk.content.lower()
        assert "subscription" in top or "cancel" in top

    def test_nested_directory(self, temp_dir, embedder):
        """Ingest nested directory structure."""
        root = temp_dir / "nested"
        root.mkdir()
        (root / "top.txt").write_text("Top level document about billing.")
        sub = root / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("Nested document about account settings.")

        kb = LocalKB(name="nested-test", path=str(temp_dir), embedder=embedder, persist=False)
        count = kb.add(str(root))
        assert count >= 2


# ---------------------------------------------------------------------------
# Multi-KB domain sharding
# ---------------------------------------------------------------------------


class TestMultiKBIntegration:
    def test_domain_sharded_search(self, temp_dir, embedder):
        """Each KB only searches its own domain."""
        kb_billing = LocalKB(
            name="billing", path=str(temp_dir), embedder=embedder, persist=False,
        )
        kb_billing.add(SUPPORT_DOCS["refund_policy"])

        kb_tech = LocalKB(
            name="tech", path=str(temp_dir), embedder=embedder, persist=False,
        )
        kb_tech.add(SUPPORT_DOCS["error_codes"])

        # Billing KB should find refund info, not error codes
        billing_results = kb_billing.search("ERR-4012", top_k=3)
        for r in billing_results:
            assert "ERR-4012" not in r.chunk.content

        # Tech KB should find error codes
        tech_results = kb_tech.search("ERR-4012", top_k=1)
        assert len(tech_results) >= 1
        assert "ERR-4012" in tech_results[0].chunk.content


# ---------------------------------------------------------------------------
# as_tool integration
# ---------------------------------------------------------------------------


class TestToolIntegration:
    def test_as_tool_with_real_embeddings(self, temp_dir, embedder):
        """KB tool works with real embeddings."""
        kb = LocalKB(
            name="tool-test", path=str(temp_dir), embedder=embedder,
            persist=False, chunk_size=256,
        )
        kb.add(SUPPORT_DOCS["faq"])

        tool = kb.as_tool()
        assert tool.name == "search_tool-test"

        # Tool returns formatted string with top results
        result = tool.execute({"query": "student discount", "top_k": 5})
        assert result.success
        assert "Score" in result.output

        # Verify via direct search that student chunk is findable
        search_results = kb.search("student discount", top_k=5)
        contents = " ".join(r.chunk.content.lower() for r in search_results)
        assert "student" in contents
