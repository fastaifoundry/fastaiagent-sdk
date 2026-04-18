"""Chroma-backed ``VectorStore`` implementation.

Install with::

    pip install 'fastaiagent[chroma]'

Supports three client modes:

- **Ephemeral** (default) — in-process, in-memory. Great for tests.
- **Persistent** — on-disk at ``persist_path``.
- **HTTP** — remote Chroma server via ``host``/``port``.
"""

from __future__ import annotations

from typing import Any

from fastaiagent.kb.chunking import Chunk


class ChromaVectorStore:
    """Chroma-backed vector store.

    Args:
        collection: Chroma collection name. Auto-created on first access.
        dimension: Vector dimensionality. Chroma does not validate this —
            kept for protocol compatibility.
        persist_path: Directory for on-disk persistence. ``None`` = ephemeral.
        host: Remote Chroma server host. Mutually exclusive with ``persist_path``.
        port: Remote Chroma server port (defaults to 8000 when host is set).
    """

    def __init__(
        self,
        collection: str,
        dimension: int,
        persist_path: str | None = None,
        host: str | None = None,
        port: int = 8000,
    ):
        try:
            import chromadb
        except ImportError as err:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "ChromaVectorStore requires chromadb. "
                "Install with: pip install 'fastaiagent[chroma]'"
            ) from err

        if host and persist_path:
            raise ValueError(
                "ChromaVectorStore accepts host= OR persist_path=, not both"
            )

        if host:
            self._client = chromadb.HttpClient(host=host, port=port)
        elif persist_path:
            self._client = chromadb.PersistentClient(path=persist_path)
        else:
            self._client = chromadb.EphemeralClient()

        self._dimension = dimension
        self._collection_name = collection
        # Use l2 distance on cosine-normalized embeddings (equivalent ranking).
        self._collection = self._client.get_or_create_collection(name=collection)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _metadata_for(self, chunk: Chunk) -> dict[str, Any]:
        # Chroma metadata must be a flat dict of primitives. Spread known
        # fields and flatten nested user metadata by stringifying complex values.
        flat: dict[str, Any] = {
            "index": chunk.index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }
        for k, v in chunk.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                flat[f"m_{k}"] = v
            else:
                import json as _json

                flat[f"m_{k}"] = _json.dumps(v, default=str)
        return flat

    def _chunk_from(self, chunk_id: str, content: str, metadata: dict[str, Any]) -> Chunk:
        import json as _json

        meta: dict[str, Any] = {}
        for k, v in metadata.items():
            if k.startswith("m_"):
                key = k[2:]
                if isinstance(v, str) and v.startswith(("{", "[")):
                    try:
                        meta[key] = _json.loads(v)
                    except Exception:
                        meta[key] = v
                else:
                    meta[key] = v
        return Chunk(
            id=chunk_id,
            content=content,
            metadata=meta,
            index=int(metadata.get("index", 0) or 0),
            start_char=int(metadata.get("start_char", 0) or 0),
            end_char=int(metadata.get("end_char", 0) or 0),
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must be aligned: "
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            embeddings=[list(e) for e in embeddings],  # type: ignore[arg-type]
            metadatas=[self._metadata_for(c) for c in chunks],
        )

    def search(
        self, query_embedding: list[float], top_k: int
    ) -> list[tuple[Chunk, float]]:
        res = self._collection.query(
            query_embeddings=[list(query_embedding)],  # type: ignore[arg-type]
            n_results=top_k,
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        results: list[tuple[Chunk, float]] = []
        for cid, content, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns distance; convert to a similarity-like score.
            # For cosine distance: similarity = 1 - distance.
            score = 1.0 - float(dist) if dist is not None else 0.0
            meta_dict: dict[str, Any] = dict(meta) if meta else {}
            results.append((self._chunk_from(cid, content or "", meta_dict), score))
        return results

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self._collection.delete(ids=list(chunk_ids))

    def rebuild(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.reset()
        self.add(chunks, embeddings)

    def reset(self) -> None:
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

    def count(self) -> int:
        return int(self._collection.count())
