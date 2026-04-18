"""Storage protocols for pluggable knowledge-base backends.

The knowledge-base stack is composed of three orthogonal storage concerns:

- ``VectorStore``    — dense-vector similarity search (FAISS, Qdrant, Chroma, Weaviate)
- ``KeywordStore``   — lexical / BM25 keyword retrieval
- ``MetadataStore``  — durable document and chunk storage + metadata

Each is a structural ``typing.Protocol`` — adapters do not need to inherit
from a base class, they just need to implement the methods. A single backend
class may implement multiple protocols (e.g. a Qdrant backend can serve as
both ``VectorStore`` and ``MetadataStore`` via payload fields).

All protocols are **synchronous** to match the existing ``Embedder`` protocol
at :mod:`fastaiagent.kb.embedding`. Adapters wrapping async upstream clients
(``AsyncQdrantClient``, etc.) are expected to bridge with ``asyncio.run`` or
``anyio.from_thread`` internally.

Future work (deliberately deferred to keep 0.3.0 non-breaking):

- **Option B** — add parallel ``a*`` async methods (``aadd``, ``asearch``,
  ``aembed`` on :class:`fastaiagent.kb.embedding.Embedder`, etc.) alongside
  the sync methods, matching the ``run``/``arun`` pattern already used by
  :class:`fastaiagent.agent.Agent`. Purely additive; no breakage.
- **Option C** — migrate protocols to async-only with sync wrappers on
  ``LocalKB``. Breaks external subclasses of ``Embedder`` — requires a
  major-version bump.

Track in the project roadmap. Do not add async here without coordinating
with the ``Embedder`` protocol and the three built-in embedders
(``SimpleEmbedder``, ``FastEmbedEmbedder``, ``OpenAIEmbedder``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from fastaiagent.kb.chunking import Chunk
from fastaiagent.kb.document import Document


@runtime_checkable
class VectorStore(Protocol):
    """Dense-vector similarity store.

    Implementations key chunks by ``Chunk.id``. Embeddings are expected to be
    unit-normalized (cosine-equivalent) since the built-in ``FaissVectorStore``
    uses inner-product indexes; backends that accept any distance metric
    should document the assumption they make.
    """

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""
        ...

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert chunks and their embeddings. Must be aligned index-wise."""
        ...

    def search(
        self, query_embedding: list[float], top_k: int
    ) -> list[tuple[Chunk, float]]:
        """Return the top-``top_k`` (chunk, score) pairs, highest score first."""
        ...

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by id. Unknown ids are silently ignored."""
        ...

    def rebuild(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Replace the entire index from scratch."""
        ...

    def reset(self) -> None:
        """Drop all data from the store."""
        ...

    def count(self) -> int:
        """Number of chunks currently in the store."""
        ...


@runtime_checkable
class KeywordStore(Protocol):
    """Lexical / keyword retrieval store.

    Scores are backend-defined (e.g. BM25). Implementations must tolerate
    empty corpora and empty queries without raising.
    """

    def add(self, chunks: list[Chunk]) -> None:
        """Insert chunks."""
        ...

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Return the top-``top_k`` (chunk, score) pairs, highest score first."""
        ...

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by id. Unknown ids are silently ignored."""
        ...

    def rebuild(self, chunks: list[Chunk]) -> None:
        """Replace the entire index from scratch."""
        ...

    def reset(self) -> None:
        """Drop all data from the store."""
        ...


@runtime_checkable
class MetadataStore(Protocol):
    """Durable storage of the canonical document/chunk records.

    Separate from ``VectorStore`` and ``KeywordStore`` because the persisted
    record (content, metadata, source) is the source of truth. Vector and
    keyword stores can be rebuilt from here.
    """

    def put_document(self, doc: Document) -> None:
        """Upsert a document."""
        ...

    def get_document(self, doc_id: str) -> Document | None:
        """Look up a document by id; return None if missing."""
        ...

    def list_documents(self) -> list[Document]:
        """Return all stored documents."""
        ...

    def delete_document(self, doc_id: str) -> None:
        """Remove a document."""
        ...

    def put_chunks(self, chunks: list[Chunk], embeddings: list[list[float]] | None) -> None:
        """Upsert chunks, optionally with aligned embeddings."""
        ...

    def get_chunks(self) -> tuple[list[Chunk], list[list[float]]]:
        """Return all chunks and the embeddings aligned to them.

        If a chunk has no stored embedding, its slot in the embedding list
        is an empty list — callers must check.
        """
        ...

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Remove chunks by id."""
        ...

    def reset(self) -> None:
        """Drop all data from the store."""
        ...

    def close(self) -> None:
        """Release any underlying resources (e.g. database connections)."""
        ...
