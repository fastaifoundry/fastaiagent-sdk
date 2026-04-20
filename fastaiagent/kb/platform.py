"""PlatformKB — remote knowledge base hosted on the FastAIAgent platform.

Protocol-compatible with ``LocalKB.search()`` at the public-method level, so an
``Agent`` can accept either. Retrieval runs on the platform (hybrid search,
reranking, relevance gate — whatever the KB is configured for). The SDK is a
thin client; no local indexes, no embedders, no storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastaiagent._platform.api import get_platform_api
from fastaiagent.kb.chunking import Chunk
from fastaiagent.kb.search import SearchResult

if TYPE_CHECKING:
    from fastaiagent.tool.base import Tool


class PlatformKB:
    """Query a KB hosted on the FastAIAgent platform.

    Requires ``fa.connect(api_key=...)`` to have been called. The API key must
    have the ``kb:read`` scope and domain access to the KB.

    Example::

        import fastaiagent as fa
        fa.connect(api_key="...")

        kb = fa.PlatformKB(kb_id="kb_abc123")
        results = kb.search("refund policy", top_k=5)
        for r in results:
            print(r.score, r.chunk.content[:80])
    """

    def __init__(self, kb_id: str):
        if not kb_id:
            raise ValueError("kb_id is required")
        self.kb_id = kb_id
        self.name = kb_id  # parity with LocalKB.name

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        from fastaiagent.kb._tracing import retrieval_span

        with retrieval_span(
            kb_name=self.name,
            backend="platform",
            search_type=None,
            query=query,
            top_k=top_k,
        ) as span:
            api = get_platform_api()
            response = api.post(
                f"/public/v1/knowledge-bases/{self.kb_id}/search",
                {"query": query, "top_k": top_k},
            )
            results = self._parse_results(response.get("results", []))
            span.record(results)
            return results

    async def asearch(self, query: str, top_k: int = 5) -> list[SearchResult]:
        from fastaiagent.kb._tracing import retrieval_span

        with retrieval_span(
            kb_name=self.name,
            backend="platform",
            search_type=None,
            query=query,
            top_k=top_k,
        ) as span:
            api = get_platform_api()
            response = await api.apost(
                f"/public/v1/knowledge-bases/{self.kb_id}/search",
                {"query": query, "top_k": top_k},
            )
            results = self._parse_results(response.get("results", []))
            span.record(results)
            return results

    def as_tool(self) -> Tool:
        """Create a FunctionTool that wraps this KB for agent use.

        Mirrors ``LocalKB.as_tool()`` so a ``PlatformKB`` can drop into the same
        ``tools=[kb.as_tool()]`` wiring on ``Agent``.
        """
        from fastaiagent.tool.function import FunctionTool

        def kb_search(query: str, top_k: int = 5) -> str:
            results = self.search(query, top_k=top_k)
            if not results:
                return "No results found."
            parts = []
            for r in results:
                parts.append(f"[Score: {r.score:.3f}] {r.chunk.content[:200]}")
            return "\n\n".join(parts)

        return FunctionTool(
            name=f"search_{self.name}",
            fn=kb_search,
            description=f"Search the '{self.name}' platform knowledge base",
        )

    @staticmethod
    def _parse_results(items: list[dict[str, Any]]) -> list[SearchResult]:
        return [
            SearchResult(
                chunk=Chunk(
                    id=item["chunk_id"],
                    content=item["content"],
                    metadata={
                        **(item.get("metadata") or {}),
                        "document_name": item.get("document_name"),
                        "document_id": item.get("document_id"),
                        "source_type": item.get("source_type"),
                    },
                    index=0,
                    start_char=0,
                    end_char=0,
                ),
                score=float(item["score"]),
            )
            for item in items
        ]
