"""Qdrant-backed ``VectorStore`` implementation.

Install with::

    pip install 'fastaiagent[qdrant]'

For local testing pass ``location=":memory:"`` to use an ephemeral
in-process Qdrant instance:

    QdrantVectorStore(location=":memory:", collection="demo", dimension=384)
"""

from __future__ import annotations

from typing import Any

from fastaiagent.kb.chunking import Chunk


class QdrantVectorStore:
    """Qdrant-backed vector store.

    Stores chunk ``content`` and ``metadata`` in the point payload so a
    single Qdrant collection can round-trip everything the protocol needs
    — no separate metadata store required for vector-only use cases.

    Args:
        collection: Qdrant collection name. Auto-created on first access.
        dimension: Vector dimensionality. Required for auto-create.
        url: HTTP URL of a remote Qdrant. Mutually exclusive with ``location``.
        location: Pass ``":memory:"`` for an in-process ephemeral instance.
        api_key: Optional API key for Qdrant Cloud.
        distance: ``"Cosine"`` (default), ``"Dot"``, or ``"Euclid"``.
    """

    def __init__(
        self,
        collection: str,
        dimension: int,
        url: str | None = None,
        location: str | None = None,
        api_key: str | None = None,
        distance: str = "Cosine",
    ):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as err:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "QdrantVectorStore requires qdrant-client. "
                "Install with: pip install 'fastaiagent[qdrant]'"
            ) from err

        if url is None and location is None:
            raise ValueError("QdrantVectorStore requires either url= or location=")
        if url and location:
            raise ValueError("QdrantVectorStore accepts url= OR location=, not both")

        self._models = models
        self._collection = collection
        self._dimension = dimension
        self._distance_name = distance
        if location:
            self._client = QdrantClient(location=location)
        else:
            self._client = QdrantClient(url=url, api_key=api_key)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            return
        distance = getattr(self._models.Distance, self._distance_name.upper())
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=self._models.VectorParams(
                size=self._dimension,
                distance=distance,
            ),
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def _chunk_to_payload(self, chunk: Chunk) -> dict[str, Any]:
        return {
            "content": chunk.content,
            "metadata": chunk.metadata,
            "index": chunk.index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }

    def _payload_to_chunk(self, chunk_id: str, payload: dict[str, Any]) -> Chunk:
        return Chunk(
            id=chunk_id,
            content=payload.get("content", ""),
            metadata=payload.get("metadata", {}) or {},
            index=payload.get("index", 0) or 0,
            start_char=payload.get("start_char", 0) or 0,
            end_char=payload.get("end_char", 0) or 0,
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must be aligned: "
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        points = [
            self._models.PointStruct(
                id=c.id,
                vector=list(emb),
                payload=self._chunk_to_payload(c),
            )
            for c, emb in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def search(
        self, query_embedding: list[float], top_k: int
    ) -> list[tuple[Chunk, float]]:
        # qdrant-client >= 1.10 prefers ``query_points``; fall back to
        # ``search`` for older versions.
        query_fn = getattr(self._client, "query_points", None)
        if query_fn is not None:
            response = query_fn(
                collection_name=self._collection,
                query=list(query_embedding),
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        else:  # pragma: no cover — exercised only on qdrant-client < 1.10
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=list(query_embedding),
                limit=top_k,
                with_payload=True,
            )
        return [
            (self._payload_to_chunk(str(h.id), h.payload or {}), float(h.score))
            for h in hits
        ]

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self._client.delete(
            collection_name=self._collection,
            points_selector=self._models.PointIdsList(points=list(chunk_ids)),
            wait=True,
        )

    def rebuild(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.reset()
        self.add(chunks, embeddings)

    def reset(self) -> None:
        self._client.delete_collection(collection_name=self._collection)
        self._ensure_collection()

    def count(self) -> int:
        result = self._client.count(collection_name=self._collection, exact=True)
        return int(result.count)
