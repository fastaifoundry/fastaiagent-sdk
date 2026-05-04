"""Phase 9 — KB consumption helpers (real KB, real LLM where needed).

Spec test IDs covered: #29 (LangChain retriever returns docs), #30
(CrewAI tool callable returns docs), #31 (PydanticAI tool callable
returns docs).

Each test seeds a small in-process LocalKB (keyword search — no
fastembed dependency) and exercises the harness wrapper. Only #29
needs an LLM call when we actually plug the retriever into a chain;
the basic "retrieves documents" assertion runs cold.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))

needs_openai = pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")

pytestmark = pytest.mark.e2e


_SEED_DOCS = [
    "Refund Policy v4.2: Customers may request a full refund within 30 days of purchase.",
    "Shipping Policy: Standard shipping takes 3-5 business days. International orders ship via DHL.",
    "Warranty: All electronics carry a one-year limited warranty covering manufacturing defects.",
]


@pytest.fixture
def seeded_kb(tmp_path: Path) -> tuple[str, Path]:
    """Spin up a keyword-only LocalKB with a few docs and return its
    name + path so harness helpers can re-open it."""
    from fastaiagent import LocalKB

    kb_name = f"harness-kb-{uuid.uuid4().hex[:8]}"
    kb = LocalKB(
        name=kb_name,
        path=str(tmp_path),
        search_type="keyword",
        persist=True,
    )
    try:
        for doc in _SEED_DOCS:
            kb.add(doc)
    finally:
        kb.close()
    return kb_name, tmp_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _override_kb_path(monkeypatch, kb_path: Path) -> None:
    """Point ``LocalKB(name=...)`` at our temp path for the duration of
    the test by monkey-patching the default path argument.

    Phase 9 helpers construct ``LocalKB(name=kb_name)`` with the default
    path; this monkey-patch makes them resolve to the seeded location
    without changing the helper signatures (the spec doesn't include a
    ``path=`` parameter on ``kb_as_retriever`` / ``kb_as_tool``).
    """
    from fastaiagent.kb import local as kb_local

    original_init = kb_local.LocalKB.__init__

    def patched_init(self, name="default", path=None, **kw):  # type: ignore[no-untyped-def]
        if path is None:
            path = str(kb_path)
        return original_init(self, name=name, path=path, **kw)

    monkeypatch.setattr(kb_local.LocalKB, "__init__", patched_init)


# ---------------------------------------------------------------------------
# #29 — LangChain retriever
# ---------------------------------------------------------------------------


def test_29_langchain_retriever_returns_docs(seeded_kb, monkeypatch) -> None:
    from fastaiagent.integrations import langchain as lc

    kb_name, kb_path = seeded_kb
    _override_kb_path(monkeypatch, kb_path)

    retriever = lc.kb_as_retriever(kb_name, top_k=3)
    docs = retriever.invoke("refund within 30 days")
    assert docs, "retriever returned zero docs"
    assert any("refund" in d.page_content.lower() for d in docs)
    # Score is propagated through the metadata.
    assert all("score" in d.metadata for d in docs)


@needs_openai
def test_29b_langchain_retriever_in_chain(seeded_kb, monkeypatch) -> None:
    """Plug the retriever into an LCEL chain to confirm it satisfies the
    LangChain ``BaseRetriever`` interface end-to-end."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI

    from fastaiagent.integrations import langchain as lc

    kb_name, kb_path = seeded_kb
    _override_kb_path(monkeypatch, kb_path)

    retriever = lc.kb_as_retriever(kb_name, top_k=3)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer using the context. If unsure, say you don't know."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    )
    out = chain.invoke("How many days do customers have to request a refund?")
    assert "30" in str(out.content), out.content


# ---------------------------------------------------------------------------
# #30 — CrewAI tool
# ---------------------------------------------------------------------------


def test_30_crewai_kb_as_tool_returns_docs(seeded_kb, monkeypatch) -> None:
    from fastaiagent.integrations import crewai as ca

    kb_name, kb_path = seeded_kb
    _override_kb_path(monkeypatch, kb_path)

    tool = ca.kb_as_tool(kb_name, top_k=3, description="Search policies.")
    # CrewAI BaseTool exposes ``run`` for callers; ``_run`` is the
    # implementation hook.
    out = tool._run("refund within 30 days")
    assert isinstance(out, str)
    assert "refund" in out.lower()
    assert "score=" in out


# ---------------------------------------------------------------------------
# #31 — PydanticAI tool
# ---------------------------------------------------------------------------


def test_31_pydanticai_kb_as_tool_returns_docs(seeded_kb, monkeypatch) -> None:
    from fastaiagent.integrations import pydanticai as pa

    kb_name, kb_path = seeded_kb
    _override_kb_path(monkeypatch, kb_path)

    tool_fn = pa.kb_as_tool(kb_name, top_k=3)
    # The returned callable is a plain function suitable for
    # Agent(tools=[fn]) / @agent.tool_plain.
    assert callable(tool_fn)
    assert tool_fn.__name__ == f"search_{kb_name}"

    out = tool_fn("refund within 30 days")
    assert isinstance(out, str)
    assert "refund" in out.lower()
