"""Example 28: LocalKB backed by Chroma.

Runs entirely in-process using Chroma's ephemeral client — no server, no
network, no API keys needed. The same adapter works against on-disk Chroma
(pass ``persist_path=...``) and remote Chroma (pass ``host=..., port=...``).

Install:
    pip install 'fastaiagent[chroma]'

Usage:
    python examples/28_kb_chroma.py
"""

from __future__ import annotations

from fastaiagent.kb import LocalKB
from fastaiagent.kb.backends.chroma import ChromaVectorStore
from fastaiagent.kb.embedding import SimpleEmbedder


def main() -> None:
    # 384-dim matches most sentence-transformers; SimpleEmbedder is a toy
    # embedder used here to keep the example offline. For real use, swap in
    # FastEmbedEmbedder or OpenAIEmbedder with matching dimension.
    dim = 384
    embedder = SimpleEmbedder(dimensions=dim)

    kb = LocalKB(
        name="chroma-demo",
        search_type="vector",
        embedder=embedder,
        vector_store=ChromaVectorStore(collection="chroma_demo", dimension=dim),
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
