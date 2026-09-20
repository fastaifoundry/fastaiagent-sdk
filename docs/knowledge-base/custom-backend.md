# Writing a Custom KB Backend

The three storage protocols — `VectorStore`, `KeywordStore`, `MetadataStore` — are [structural `typing.Protocol`s](https://peps.python.org/pep-0544/). You don't inherit from a base class; you just implement the methods with the right signatures. This page walks through a minimal `VectorStore` adapter.

## When to write your own

- You're on an existing vector DB we don't ship an adapter for yet (pgvector, Weaviate, Milvus, Pinecone, Elasticsearch, Redis, OpenSearch, …)
- You want cache / replication / auth / tracing behavior a generic adapter can't cover
- You're wrapping a proprietary or internal search service

## Minimal `VectorStore` skeleton

```python
from fastaiagent.kb.chunking import Chunk


class MyVectorStore:
    def __init__(self, dimension: int):
        self._store: dict[str, tuple[Chunk, list[float]]] = {}
        self._dim = dimension

    # --- required protocol methods ---

    @property
    def dimension(self) -> int:
        return self._dim

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be aligned")
        for c, e in zip(chunks, embeddings):
            self._store[c.id] = (c, list(e))

    def search(self, query_embedding, top_k):
        # compute similarity however your backend does it
        scored = [
            (c, self._cosine(query_embedding, emb))
            for c, emb in self._store.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete(self, chunk_ids):
        for cid in chunk_ids:
            self._store.pop(cid, None)

    def rebuild(self, chunks, embeddings):
        self._store.clear()
        self.add(chunks, embeddings)

    def reset(self):
        self._store.clear()

    def count(self):
        return len(self._store)
```

That's it — pass an instance to `LocalKB`:

```python
kb = LocalKB(
    name="custom",
    vector_store=MyVectorStore(dimension=384),
)
```

## What `LocalKB` guarantees when it calls you

- **`add(chunks, embeddings)`** — `len(chunks) == len(embeddings)`, every `chunks[i]` has a unique `id`, every `embeddings[i]` has length `self.dimension`. Called once per `add_documents` batch.
- **`search(query_embedding, top_k)`** — `top_k >= 1`. Return an empty list if the store is empty; do not raise.
- **`delete(chunk_ids)`** — chunk ids may or may not exist; unknown ids should be silently ignored.
- **`rebuild(chunks, embeddings)`** — full replacement. `LocalKB` calls this after `delete` / `update` to keep indexes in sync when the backend does not support per-id delete efficiently.
- **`reset()`** — total wipe.
- **`count()`** — approximate is fine if exact is expensive; document the behavior in your adapter's docstring.

## Contract tests

Run your adapter against the same contract test suite used for the built-in backends:

```python
# tests/test_my_vector_store.py
from tests.test_kb_protocols import VectorStoreContract

class TestMyVectorStoreContract(VectorStoreContract):
    def make(self):
        return MyVectorStore(...)
```

That single subclass runs the full contract — add/search roundtrip, delete, rebuild, reset, misaligned-args error — against your adapter.

## `KeywordStore` and `MetadataStore`

`KeywordStore` is the same shape minus the embedding arg and the `dimension` property. `MetadataStore` adds document-level operations (`put_document`, `list_documents`, `get_chunks`, etc.). See [`fastaiagent/kb/protocols.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/fastaiagent/kb/protocols.py) for the full signatures.

## Tips

- **Score semantics** — `search` must return highest-score-first **and** return a cosine similarity in `[-1, 1]`: `1.0` identical, `0.0` orthogonal, `-1.0` opposite. The value is part of the contract, not just the order — `VectorBlock` fuses it with recency as a weighted sum, `LocalKB`'s hybrid mode hands raw vector scores straight back when the keyword side is empty, and `LocalKB.as_tool()` prints the number into the text the model reads.
- **Converting a distance** — there is no safe default. `similarity = 1 - distance` is right **only** for a true cosine distance. Check the space your index is actually configured with, and convert from that one: squared L2 (what Chroma's default `l2` reports, and FAISS's `IndexFlatL2`) needs `1 - d/2` for unit vectors, plain Euclidean needs `1 - d²/2`, and an inner-product index usually needs no conversion at all. Assuming cosine over a squared-L2 distance is exactly the defect `ChromaVectorStore` shipped with until 1.67.0 — it returned `2·cos - 1`, down to `-3.0`, and no test caught it because every test asserted the score was a `float` and none asserted a value.
- **Test the values** — `tests/test_kb_score_semantics_sweep.py` is the cross-backend sweep: identical `≈ 1.0`, orthogonal `≈ 0.0`, opposite `≈ -1.0`. Add your backend to its factory map.
- **Metadata round-trip** — `Chunk.metadata` is a free-form `dict[str, Any]`. Serialize non-primitive values when the backend requires it (JSON-encode + parse back on read — see `ChromaVectorStore` for an example).
- **Lazy connections** — if your backend is remote, hold off on opening the connection until the first `add`/`search`. Tests construct your adapter many times; make that cheap.
- **Don't raise on unknown ids in delete** — silent no-op is the contract.

## Share it back

If you build a useful adapter, please open a PR. We want the `fastaiagent/kb/backends/` directory to grow.

---

## Next Steps

- [Backends Overview](backends.md) — Shipping adapters and usage patterns
- [Knowledge Base Overview](index.md) — Full `LocalKB` reference
