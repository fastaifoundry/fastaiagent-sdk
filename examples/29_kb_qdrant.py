"""Example 29: LocalKB backed by Qdrant.

Uses Qdrant's in-process ``location=":memory:"`` client — no server required.
Swap to ``url="http://localhost:6333"`` (or Qdrant Cloud) without code changes
elsewhere.

Install:
    pip install 'fastaiagent[qdrant]'

Usage:
    python examples/29_kb_qdrant.py
"""

from __future__ import annotations

from fastaiagent.kb import LocalKB
from fastaiagent.kb.backends.qdrant import QdrantVectorStore
from fastaiagent.kb.embedding import SimpleEmbedder


def main() -> None:
    dim = 384
    embedder = SimpleEmbedder(dimensions=dim)

    kb = LocalKB(
        name="qdrant-demo",
        search_type="vector",
        embedder=embedder,
        vector_store=QdrantVectorStore(
            collection="qdrant_demo",
            dimension=dim,
            location=":memory:",
        ),
        persist=False,
    )

    kb.add("The mitochondria is the powerhouse of the cell.")
    kb.add("Octopuses have three hearts and blue blood.")
    kb.add("The Pacific Ocean is the largest body of water on Earth.")

    for query in ["cell biology", "deep sea", "octopus"]:
        results = kb.search(query, top_k=2)
        print(f"\nQuery: {query!r}")
        for r in results:
            print(f"  [{r.score:.3f}] {r.chunk.content}")

    print(f"\nStatus: {kb.status()}")


if __name__ == "__main__":
    main()
