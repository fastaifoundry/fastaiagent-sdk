# Pluggable KB Backends

Since 0.3.0, `LocalKB` is composed of three orthogonal storage layers that can be swapped independently:

| Layer | Protocol | Default | Pluggable options |
|---|---|---|---|
| Vector search | `VectorStore` | `FaissVectorStore` (in-process) | `QdrantVectorStore`, `ChromaVectorStore`, your own |
| Keyword search | `KeywordStore` | `BM25KeywordStore` (in-process) | your own |
| Document/chunk storage | `MetadataStore` | `SqliteMetadataStore` (on-disk) | your own |

Default behavior — `LocalKB(name="docs")` with no kwargs — is **byte-for-byte identical** to 0.2.x. You only need to touch backends when you want to point at an existing Qdrant/Chroma deployment, scale beyond one machine, or write your own storage adapter.

## The Three Protocols

All protocols are **synchronous** (matching the existing `Embedder` protocol). See [`fastaiagent/kb/protocols.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/fastaiagent/kb/protocols.py) for the canonical definitions.

```python
from fastaiagent.kb import VectorStore, KeywordStore, MetadataStore

class VectorStore(Protocol):
    @property
    def dimension(self) -> int: ...
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]: ...
    def delete(self, chunk_ids: list[str]) -> None: ...
    def rebuild(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def reset(self) -> None: ...
    def count(self) -> int: ...
```

`KeywordStore` mirrors this minus the embedding arg; `MetadataStore` adds document-level operations (`put_document`, `list_documents`, `get_chunks`, etc.).

## Default Backends

You already use these — no code change needed.

- **`FaissVectorStore`** — wraps the existing `FaissIndex` (flat/ivf/hnsw). In-process, no services, up to ~1M vectors.
- **`BM25KeywordStore`** — wraps the existing pure-Python BM25 index. Zero dependencies.
- **`SqliteMetadataStore`** — wraps the existing SQLite document+chunk table. On-disk persistence.

Explicit construction is available if you need to share a store across KBs or wire it differently:

```python
from fastaiagent.kb import LocalKB
from fastaiagent.kb.backends.faiss import FaissVectorStore
from fastaiagent.kb.backends.bm25 import BM25KeywordStore
from fastaiagent.kb.backends.sqlite import SqliteMetadataStore

kb = LocalKB(
    name="docs",
    vector_store=FaissVectorStore(dimension=1536, index_type="hnsw"),
    keyword_store=BM25KeywordStore(k1=1.2, b=0.75),
    metadata_store=SqliteMetadataStore("/shared/storage/kb.sqlite"),
)
```

## Qdrant

Install with:

```bash
pip install 'fastaiagent[qdrant]'
```

**Remote Qdrant (self-hosted or Qdrant Cloud):**

```python
from fastaiagent.kb import LocalKB
from fastaiagent.kb.backends.qdrant import QdrantVectorStore
from fastaiagent.kb.embedding import OpenAIEmbedder

kb = LocalKB(
    name="product-docs",
    search_type="vector",
    embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    vector_store=QdrantVectorStore(
        url="http://localhost:6333",
        collection="product-docs",
        dimension=1536,
    ),
)

kb.add("docs/")
results = kb.search("refund policy", top_k=5)
```

**Qdrant Cloud:**

```python
vector_store=QdrantVectorStore(
    url="https://xyz.eu-central-1.aws.cloud.qdrant.io",
    api_key=os.environ["QDRANT_API_KEY"],
    collection="product-docs",
    dimension=1536,
)
```

**In-memory (tests, quick experiments):**

```python
vector_store=QdrantVectorStore(
    location=":memory:",
    collection="scratch",
    dimension=384,
)
```

Qdrant stores chunk content and metadata in the point payload, so a single Qdrant collection round-trips everything `VectorStore` needs. The default distance is cosine — pass `distance="Dot"` or `"Euclid"` to override.

## Chroma

Install with:

```bash
pip install 'fastaiagent[chroma]'
```

**Ephemeral (in-process, in-memory — great for tests):**

```python
from fastaiagent.kb.backends.chroma import ChromaVectorStore

kb = LocalKB(
    name="scratch",
    search_type="vector",
    vector_store=ChromaVectorStore(collection="scratch", dimension=384),
    persist=False,
)
```

**Persistent (on-disk):**

```python
vector_store=ChromaVectorStore(
    collection="product-docs",
    dimension=1536,
    persist_path="/var/lib/fastaiagent/chroma",
)
```

**Remote Chroma server:**

```python
vector_store=ChromaVectorStore(
    collection="product-docs",
    dimension=1536,
    host="chroma.internal",
    port=8000,
)
```

Chroma metadata is a flat primitive-typed dict — the adapter flattens nested `Chunk.metadata` values by JSON-encoding non-primitive fields and reparsing on search. You see the same `dict` shape in and out.

## Choosing a Backend

| Scenario | Recommended |
|---|---|
| Solo dev, small KB, no infra | Default (FAISS + BM25 + SQLite) |
| Team, shared KB, fast to prototype | Default with `persist_path` on shared storage |
| Already running Chroma | `ChromaVectorStore` |
| Already running Qdrant | `QdrantVectorStore` |
| High-throughput, multi-tenant | `QdrantVectorStore` (Cloud or self-hosted cluster) |
| Tests / CI | `ChromaVectorStore(persist_path=None)` or `QdrantVectorStore(location=":memory:")` |

## Writing Your Own Backend

See [Custom Backend](custom-backend.md) for a step-by-step guide. The short version: implement the methods on `VectorStore`, `KeywordStore`, or `MetadataStore` (no base class required — they're structural `Protocol`s) and pass an instance to `LocalKB`.

## Future Work

Async (`aadd`, `asearch`, `aembed`, ...) parallel methods on all protocols are planned. They will be **additive** — nothing you write against the sync protocols today will break. See the module docstring in [`fastaiagent/kb/protocols.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/fastaiagent/kb/protocols.py) for the roadmap.

---

## Next Steps

- [Custom Backend](custom-backend.md) — Implement your own storage adapter
- [Knowledge Base Overview](index.md) — Full `LocalKB` reference
