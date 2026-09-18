"""Chroma-backed ``VectorStore`` implementation.

Install with::

    pip install 'fastaiagent[chroma]'

Supports three client modes:

- **Ephemeral** (default) — in-process, in-memory. Great for tests.
- **Persistent** — on-disk at ``persist_path``.
- **HTTP** — remote Chroma server via ``host``/``port``.

**Scores are cosine similarity in ``[-1, 1]``**, matching FAISS and Qdrant —
see :class:`fastaiagent.kb.protocols.VectorStore`. Getting there takes two
things, and having only one of them is what made every Chroma score wrong
before 1.67.0:

1. New collections are created with ``space="cosine"``.
2. The space of the collection actually in use is **detected**, and the
   distance converted from whatever it turns out to be.

Detection is not belt-and-braces. Chroma silently ignores a space passed to
``get_or_create_collection`` for a collection that already exists — no error,
no warning, the space stays whatever it was created with. So setting cosine
without detecting would leave every persisted and remote collection on ``l2``
while this module assumed cosine: the same defect with its evidence removed.

The original bug: the collection was created with no space at all, which makes
Chroma use its HNSW default of ``l2``, and ``l2`` in chromadb reports **squared**
Euclidean distance. Converting that with ``1 - d`` (the cosine formula) returns
``2·cos - 1``, a range of ``[-3, 1]``: ``-1.0`` for a truly orthogonal pair and
``-3.0`` for an opposite one. Ranking survived for unit-normalized embeddings
(the transform is monotonic), but every consumer that reads the *value* did
not — ``VectorBlock``'s recency fusion, which assumes ``[0, 1]`` and provably
inverted its ranking against a FAISS-backed store; ``LocalKB._hybrid_search``,
which hands raw vector scores back when the keyword side is empty; and
``LocalKB.as_tool()``, which formats the number into the text the model reads.

**Migrating an existing collection**: ``reset()`` genuinely drops and recreates,
so ``kb.clear()`` followed by a re-index is the one-line migration to a native
cosine collection. It is optional — an ``l2`` collection is converted correctly
and warned about once — and only matters for embedders that do not unit-
normalize, where ``l2`` and cosine rank differently.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from fastaiagent.kb.chunking import Chunk

logger = logging.getLogger(__name__)


#: Distance -> cosine-similarity conversion, per HNSW space, as chromadb 1.x
#: actually reports them (measured, not inferred from the docs):
#:
#: * ``l2``    — **squared** Euclidean. For unit vectors ``d = 2(1 - cos)``,
#:               so ``cos = 1 - d/2``. The shipped code used ``1 - d``.
#: * ``cosine`` — ``d = 1 - cos``, so ``cos = 1 - d``.
#: * ``ip``     — ``d = 1 - <a,b>``, so ``<a,b> = 1 - d``. (Not ``-d``: chromadb
#:                offsets it the same way it does cosine. Verified against
#:                magnitude-2 vectors, where a raw inner product of 4 comes back
#:                as ``-3.0``.)
_SPACE_TO_SIMILARITY = {
    "l2": lambda d: 1.0 - d / 2.0,
    "cosine": lambda d: 1.0 - d,
    "ip": lambda d: 1.0 - d,
}

#: What Chroma uses when no space is given. Also the fallback when neither
#: ``configuration_json`` nor ``metadata`` names one.
_DEFAULT_SPACE = "l2"

#: The configuration this SDK creates collections with. Passed on BOTH
#: ``__init__`` and ``reset()``: ``reset()`` drops and re-creates, so a fix that
#: only touched ``__init__`` would silently fall back to ``l2`` on the first
#: ``kb.clear()`` or ``rebuild()``.
# Typed ``Any`` rather than ``dict[str, Any]``: chromadb declares this parameter
# as its own ``CreateCollectionConfiguration`` TypedDict, which only exists on
# 1.x, and the pin admits 0.5+. A plain dict is what the API accepts at runtime
# on every version in range.
_COSINE_CONFIG: Any = {"hnsw": {"space": "cosine"}}


class ChromaVectorStore:
    """Chroma-backed vector store.

    ``search`` returns cosine similarity in ``[-1, 1]``, like every other
    built-in backend — see the module docstring for how, and why detecting the
    collection's space is load-bearing rather than defensive.

    Args:
        collection: Chroma collection name. Auto-created on first access, with
            ``space="cosine"``. An existing collection keeps whatever space it
            was created with; its distances are converted accordingly.
        dimension: Vector dimensionality. Chroma does not validate this —
            kept for protocol compatibility.
        persist_path: Directory for on-disk persistence. ``None`` = ephemeral.
        host: Remote Chroma server host. Mutually exclusive with ``persist_path``.
        port: Remote Chroma server port (defaults to 8000 when host is set).
    """

    #: Collection names already warned about, so the "this is an l2 collection"
    #: notice is one line per collection rather than one per store instance.
    _warned_spaces: ClassVar[set[str]] = set()

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
        self._collection = self._create_collection()
        self._space = self._detect_space()
        self._warn_if_not_cosine()

    def _create_collection(self) -> Any:
        """Get or create the collection, asking for cosine.

        The request is honoured for a NEW collection and silently ignored for
        one that already exists — which is exactly why ``_detect_space`` runs
        afterwards instead of this method's return value being trusted.
        """
        try:
            return self._client.get_or_create_collection(
                name=self._collection_name, configuration=_COSINE_CONFIG
            )
        except TypeError:
            # ``configuration=`` arrived mid-0.x; the pin is ``>=0.5,<2``.
            # Older clients take the space through collection metadata.
            logger.debug(
                "Chroma client does not accept configuration=; falling back to "
                "metadata={'hnsw:space': 'cosine'}",
                exc_info=True,
            )
            return self._client.get_or_create_collection(
                name=self._collection_name, metadata={"hnsw:space": "cosine"}
            )

    def _detect_space(self) -> str:
        """The HNSW space the live collection actually uses.

        Fallback chain suited to the ``chromadb>=0.5,<2`` pin:

        1. ``collection.configuration_json["hnsw"]["space"]`` — authoritative on
           1.x and present for Ephemeral, Persistent and Http alike.
        2. ``collection.metadata["hnsw:space"]`` — ``metadata`` is ``None`` for a
           default collection and carries this key only when the space was set
           that way.
        3. ``"l2"`` — Chroma's own default, and the honest assumption when
           neither source answers.
        """
        collection = self._collection
        try:
            config = getattr(collection, "configuration_json", None) or {}
            space = ((config.get("hnsw") or {}) if isinstance(config, dict) else {}).get("space")
            if isinstance(space, str) and space:
                return space
        except Exception:
            logger.debug("Chroma configuration_json unreadable", exc_info=True)
        try:
            metadata = getattr(collection, "metadata", None) or {}
            space = metadata.get("hnsw:space")
            if isinstance(space, str) and space:
                return space
        except Exception:
            logger.debug("Chroma collection metadata unreadable", exc_info=True)
        return _DEFAULT_SPACE

    def _warn_if_not_cosine(self) -> None:
        """Say once, per collection, that this one is not natively cosine."""
        if self._space == "cosine":
            return
        if self._collection_name in ChromaVectorStore._warned_spaces:
            return
        ChromaVectorStore._warned_spaces.add(self._collection_name)
        logger.warning(
            "Chroma collection %r uses the %r distance space, not cosine. Scores are "
            "converted correctly, so rankings and values are right for unit-normalized "
            "embeddings. A rebuild (kb.clear() then re-index, or store.reset()) gives a "
            "native cosine collection — worth doing if your Embedder does not normalize, "
            "because %r and cosine rank differently for vectors of varying magnitude.",
            self._collection_name,
            self._space,
            self._space,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def space(self) -> str:
        """The HNSW distance space this collection actually uses."""
        return self._space

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
                        logger.debug(
                            "Failed to parse Chroma metadata JSON for key %r", key, exc_info=True,
                        )
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
            embeddings=[list(e) for e in embeddings],
            metadatas=[self._metadata_for(c) for c in chunks],
        )

    def search(
        self, query_embedding: list[float], top_k: int
    ) -> list[tuple[Chunk, float]]:
        res = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        results: list[tuple[Chunk, float]] = []
        convert = _SPACE_TO_SIMILARITY.get(self._space)
        if convert is None:
            # A space chromadb grew after this module was written. Refuse to
            # guess: a wrong conversion is the defect this whole module now
            # documents, and a silent wrong number is worse than a loud stop.
            raise ValueError(
                f"Chroma collection {self._collection_name!r} uses the unrecognised "
                f"distance space {self._space!r}; fastaiagent knows how to convert "
                f"{sorted(_SPACE_TO_SIMILARITY)}. Recreate the collection with "
                f"space='cosine' (store.reset()) or open an issue."
            )
        for cid, content, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns a DISTANCE whose meaning depends on the collection's
            # space — squared-L2 by default, not cosine. See _SPACE_TO_SIMILARITY.
            score = convert(float(dist)) if dist is not None else 0.0
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
            logger.debug(
                "Failed to delete Chroma collection %r during reset",
                self._collection_name, exc_info=True,
            )
        # Re-created through ``_create_collection`` so the space is asked for
        # here too. Before 1.67.0 this line passed only ``name=``: a reset or a
        # rebuild dropped a cosine collection and silently made an l2 one.
        # It is also the migration lever — ``reset()`` really does drop, so
        # ``kb.clear()`` plus a re-index converts a legacy collection.
        self._collection = self._create_collection()
        ChromaVectorStore._warned_spaces.discard(self._collection_name)
        self._space = self._detect_space()
        self._warn_if_not_cosine()

    def count(self) -> int:
        return int(self._collection.count())
