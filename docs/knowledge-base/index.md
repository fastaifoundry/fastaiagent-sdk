# Knowledge Base

LocalKB is a production-ready, built-in knowledge base with FAISS vector search, BM25 keyword search, hybrid search, SQLite persistence, and full CRUD operations. No external infrastructure required.

> **New in 0.3.0 — Pluggable backends.** The vector, keyword, and metadata stores behind `LocalKB` are now swappable. Default behavior (FAISS + BM25 + SQLite) is unchanged. Point at a remote **Qdrant** or **Chroma** instance with a single kwarg. See [Backends](backends.md).

> **Hosted KBs.** For KBs uploaded and managed on the FastAIAgent platform, use [`PlatformKB`](platform-kb.md) — same `search()` surface, retrieval runs on the platform.

## Quick Start

```python
from fastaiagent.kb import LocalKB

kb = LocalKB(name="product-docs")

# Add content (text, files, or directories)
kb.add("Refund policy: Returns accepted within 30 days of purchase.")
kb.add("/path/to/faq.md")
kb.add("docs/")  # recursively ingests .txt, .md, .pdf

# Search (hybrid by default — combines FAISS + BM25)
results = kb.search("How do I return an item?", top_k=3)
for r in results:
    print(f"[{r.score:.3f}] {r.chunk.content[:80]}...")
```

Restart the process — your data is still there. No re-embedding.

## Adding Content

### Raw Text

```python
kb.add("Any text content. It will be chunked and embedded automatically.")
```

### Files

Supported formats: `.txt`, `.md`, `.pdf` (requires `pip install fastaiagent[kb]`)

```python
kb.add("docs/readme.md")
kb.add("docs/manual.pdf")      # Requires pymupdf
kb.add("docs/notes.txt")
```

### Directories

Recursively ingests all supported files:

```python
kb.add("docs/")  # Scans for .txt, .md, .pdf recursively
```

### Multiple Documents

```python
from fastaiagent.kb.document import Document

docs = [
    Document(content="First document content", source="doc1.md", metadata={"type": "faq"}),
    Document(content="Second document content", source="doc2.md", metadata={"type": "guide"}),
]
count = kb.add_documents(docs)
print(f"Added {count} chunks")
```

Each file is:
1. Read and extracted to text (PDF pages extracted individually)
2. Split into chunks (default 512 characters with 50-character overlap)
3. Embedded into vectors (skipped for `search_type="keyword"`)
4. Stored in SQLite (if `persist=True`) and indexed for search

## Searching

### Search Types

LocalKB supports three search modes. Choose based on your query patterns:

| Search Type | How It Works | Best For |
|------------|-------------|----------|
| `"vector"` | FAISS semantic similarity | Natural language queries ("how do I get a refund") |
| `"keyword"` | BM25 term matching | Exact terms, codes, IDs ("ERR-4012", "TXN-88421") |
| `"hybrid"` (default) | Vector + BM25 combined | Real-world queries mixing both ("ERR-4012 payment not working") |

```python
# Hybrid (default) — best of both worlds
kb = LocalKB(name="support-docs")

# Vector only — semantic search
kb = LocalKB(name="docs", search_type="vector")

# Keyword only — no embedder needed, zero embedding cost
kb = LocalKB(name="logs", search_type="keyword")
```

**Keyword mode** is especially useful when you don't need semantic search — it skips embedding entirely, meaning no embedder is initialized, no API calls, and instant ingestion.

### Hybrid Search and Alpha Tuning

In hybrid mode, results from FAISS and BM25 are normalized and combined:

```
final_score = alpha * vector_score + (1 - alpha) * bm25_score
```

```python
# Semantic-heavy (default) — good for most cases
kb = LocalKB(name="docs", alpha=0.7)

# Equal weight — queries mix codes + natural language
kb = LocalKB(name="docs", alpha=0.5)

# Keyword-heavy — technical docs with lots of IDs/codes
kb = LocalKB(name="docs", alpha=0.3)
```

### Searching

```python
results = kb.search("refund policy", top_k=5)

for r in results:
    print(f"Score: {r.score:.3f}")
    print(f"Content: {r.chunk.content}")
    print(f"Source: {r.chunk.metadata.get('source', 'unknown')}")
    print()
```

### Empty KB

Searching an empty KB returns an empty list — no error:

```python
kb = LocalKB(name="empty")
results = kb.search("anything")
print(len(results))  # 0
```

## FAISS Index Types

LocalKB uses FAISS for vector search. Three index types are available:

| Index Type | Algorithm | Accuracy | Speed | When to Use |
|-----------|-----------|----------|-------|-------------|
| `"flat"` (default) | Brute-force inner product | Exact (100%) | O(N) per query | Up to ~100K chunks. No tuning needed. Start here. |
| `"ivf"` | Inverted file index | ~95-99% (approximate) | Sublinear | 100K-1M chunks. Trades small accuracy for faster search. |
| `"hnsw"` | Hierarchical Navigable Small World graph | ~99% (approximate) | Very fast | Large KBs needing both speed and high recall. Uses more memory. |

```python
# Default — exact search, no config needed
kb = LocalKB(name="docs")

# Large KB — use IVF for faster approximate search
kb = LocalKB(name="big-docs", index_type="ivf")

# Speed-critical — HNSW for fastest recall
kb = LocalKB(name="realtime", index_type="hnsw")
```

Start with `"flat"` (the default). You only need `"ivf"` or `"hnsw"` if search latency becomes noticeable — typically above 100K chunks.

## Persistence

By default, LocalKB persists all data to SQLite. Chunks and embeddings survive process restarts — no re-embedding needed.

```python
# Persistent (default)
kb = LocalKB(name="docs")
kb.add("important content")
# Data saved to .fastaiagent/kb/docs/kb.sqlite

# After restart:
kb = LocalKB(name="docs")
print(kb.status()["chunk_count"])  # Still there!
```

### Temporary KB

For throwaway use cases (single agent run, dynamic API content, testing):

```python
kb = LocalKB(name="scratch", persist=False)
kb.add("temporary content from an API call")
results = kb.search("keyword")
# No files created, data gone when process ends
```

### Context Manager

```python
with LocalKB(name="docs") as kb:
    kb.add("content")
    results = kb.search("query")
# Database connection closed automatically
```

## Update and Delete

All CRUD operations persist to SQLite and update search indexes.

### Delete by Chunk ID

```python
kb.add("Content to delete")
chunk_id = kb._chunks[0].id
kb.delete(chunk_id)  # Returns True if found
```

### Delete by Source File

```python
kb.add("docs/faq.md")
deleted = kb.delete_by_source("docs/faq.md")
print(f"Removed {deleted} chunks")
```

### Update a Chunk

```python
kb.add("Original content")
chunk_id = kb._chunks[0].id
kb.update(chunk_id, "Updated content")  # Re-embeds automatically
```

### Clear Entire KB

```python
kb.clear()  # Removes all chunks, embeddings, and indexes
```

## Using KB as an Agent Tool

The most common pattern — give your agent access to a knowledge base:

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.kb import LocalKB

# Build the KB (persisted — only need to add once)
kb = LocalKB(name="product-docs")
kb.add("Refund policy: Returns within 30 days. Items must be in original condition.")
kb.add("Shipping: 3-5 business days domestic. Express shipping available.")
kb.add("Support hours: Monday-Friday 9am-5pm EST.")

# Create a search tool from the KB
search_tool = kb.as_tool()
# Creates a FunctionTool named "search_product-docs" that wraps kb.search()

# Give it to an agent
agent = Agent(
    name="support-bot",
    system_prompt="Use the search tool to find information before answering.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    tools=[search_tool],
)

result = agent.run("What is the refund policy?")
# Agent calls search_product-docs → gets relevant chunks → answers from them
```

The tool returns formatted search results:
```
[Score: 0.825] Refund policy: Returns within 30 days. Items must be in original condition.

[Score: 0.412] Support hours: Monday-Friday 9am-5pm EST.
```

## Multi-KB Agent Pattern

Use domain-sharded KBs with an agent that routes queries to the right KB:

```python
kb_billing = LocalKB(name="billing")
kb_billing.add("billing-docs/")

kb_shipping = LocalKB(name="shipping")
kb_shipping.add("shipping-docs/")

kb_returns = LocalKB(name="returns")
kb_returns.add("return-policy-docs/")

agent = Agent(
    name="support-agent",
    system_prompt="Search the relevant KB based on the customer's question.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    tools=[kb_billing.as_tool(), kb_shipping.as_tool(), kb_returns.as_tool()],
)
```

Each KB is searched independently — search stays fast even with thousands of documents across all domains.

## Embedding Providers

Three embedders are available, selected automatically based on what's installed:

### OpenAI Embeddings (recommended for production)

```python
from fastaiagent.kb.embedding import OpenAIEmbedder

kb = LocalKB(name="docs", embedder=OpenAIEmbedder(model="text-embedding-3-small"))
```

Requires: `pip install fastaiagent[openai]` and `OPENAI_API_KEY` env var.

Produces 1536-dimensional vectors with strong semantic understanding — "return an item" matches "refund policy" even without shared keywords.

### FastEmbed (local, no API calls)

```python
from fastaiagent.kb.embedding import FastEmbedEmbedder

kb = LocalKB(name="docs", embedder=FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5"))
```

Requires: `pip install fastaiagent[kb]`

Runs entirely on your machine. Good balance of quality and speed for development and privacy-sensitive use cases.

### SimpleEmbedder (fallback, no dependencies)

```python
from fastaiagent.kb.embedding import SimpleEmbedder

kb = LocalKB(name="docs", embedder=SimpleEmbedder(dimensions=128))
```

Character-frequency based — no external dependencies, no API calls. Works for testing but not suitable for production (no semantic understanding).

### Auto-Selection

If you don't specify an embedder, the best available one is chosen:

```python
kb = LocalKB(name="docs")
# Tries: FastEmbed → OpenAI → SimpleEmbedder (fallback)
```

| Priority | Embedder | Requires | Quality |
|----------|----------|----------|---------|
| 1st | FastEmbedEmbedder | `pip install fastaiagent[kb]` | Good |
| 2nd | OpenAIEmbedder | `pip install fastaiagent[openai]` + API key | Best |
| 3rd | SimpleEmbedder | Nothing | Testing only |

## Chunking

Documents are split into chunks before embedding. The recursive chunker tries progressively finer separators:

1. `\n\n` (paragraphs)
2. `\n` (lines)
3. `. ` (sentences)
4. ` ` (words)
5. Hard character split (last resort)

### Configuration

```python
kb = LocalKB(
    name="docs",
    chunk_size=512,      # Max characters per chunk (default: 512)
    chunk_overlap=50,    # Character overlap between chunks (default: 50)
)
```

**Chunk size guidelines:**

| Chunk Size | Best For |
|------------|----------|
| 256 | Short, precise answers (FAQ) |
| 512 | General purpose (default) |
| 1024 | Longer context, fewer chunks |

### Direct Chunking API

```python
from fastaiagent.kb.chunking import chunk_text

chunks = chunk_text(
    text="Your long document text...",
    chunk_size=512,
    overlap=50,
    metadata={"source": "readme.md"},
)

for c in chunks:
    print(f"[{c.id}] {c.start_char}-{c.end_char}: {c.content[:40]}...")
```

## Search Result

| Field | Type | Description |
|-------|------|-------------|
| `chunk` | `Chunk` | The matched chunk |
| `score` | `float` | Similarity score (higher is better) |

### Chunk

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique UUID for the chunk |
| `content` | `str` | Chunk text |
| `metadata` | `dict` | Source info, custom metadata |
| `index` | `int` | Position in the original document |
| `start_char` | `int` | Start character offset |
| `end_char` | `int` | End character offset |

## KB Status

```python
status = kb.status()
print(status)
# {
#     "name": "product-docs",
#     "chunk_count": 15,
#     "path": ".fastaiagent/kb/product-docs",
#     "persist": True,
#     "search_type": "hybrid",
#     "index_type": "flat"
# }
```

## Full Configuration Reference

```python
kb = LocalKB(
    name="docs",                        # KB name (used in path and tool name)
    path=".fastaiagent/kb/",            # Base storage directory
    embedder=FastEmbedEmbedder(),       # Embedding provider (auto-selected if omitted)
    chunk_size=512,                     # Max characters per chunk
    chunk_overlap=50,                   # Character overlap between chunks
    persist=True,                       # Save to SQLite (False for in-memory only)
    search_type="hybrid",              # "vector" | "keyword" | "hybrid"
    index_type="flat",                 # "flat" | "ivf" | "hnsw"
    alpha=0.7,                         # Vector vs BM25 weight in hybrid mode
)
```

## CLI Commands

```bash
# Check KB status
fastaiagent kb status --name product-docs

# Add a file or directory
fastaiagent kb add ./docs/readme.md --name product-docs
fastaiagent kb add ./docs/ --name product-docs

# Delete chunks from a source file
fastaiagent kb delete ./docs/old-faq.md --name product-docs

# Clear all data
fastaiagent kb clear --name product-docs
```

## Error Handling

```python
# Nonexistent path — treated as raw text, NOT an error
kb.add("/nonexistent/file.txt")
# Adds the string "/nonexistent/file.txt" as a text chunk
# To ingest a file, ensure the path exists first

# Explicit file ingestion (raises FileNotFoundError)
from fastaiagent.kb.document import ingest_file
try:
    ingest_file("/nonexistent/file.txt")
except FileNotFoundError:
    print("File not found")

# PDF without pymupdf
try:
    kb.add("document.pdf")  # Only works if document.pdf exists
except ImportError:
    print("Install pymupdf: pip install fastaiagent[kb]")

# Embedding dimension mismatch on reload
try:
    kb = LocalKB(name="docs", embedder=SimpleEmbedder(dimensions=64))
    # Fails if KB was created with a different dimension embedder
except ValueError as e:
    print(f"Dimension mismatch: {e}")
```

## Complete Example

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.kb import LocalKB
from fastaiagent.kb.embedding import OpenAIEmbedder

# 1. Build a KB with OpenAI embeddings (persisted — run once)
kb = LocalKB(
    name="company-docs",
    embedder=OpenAIEmbedder(),
    chunk_size=512,
)

# 2. Ingest files and directories
kb.add("docs/refund-policy.md")
kb.add("docs/shipping-guide.md")
kb.add("docs/faq/")               # Recursive directory ingestion
print(f"KB ready: {kb.status()['chunk_count']} chunks")

# 3. Create agent with KB tool
agent = Agent(
    name="support-bot",
    system_prompt=(
        "You are a customer support agent. Always search the knowledge base "
        "before answering. Cite the relevant information in your response."
    ),
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    tools=[kb.as_tool()],
)

# 4. Agent uses KB to answer
result = agent.run("Can I return a digital product?")
print(result.output)
```

---

## Next Steps

- [Backends](backends.md) — Pluggable vector, keyword, and metadata storage (Qdrant, Chroma, custom)
- [Agents](../agents/index.md) — Build agents that use knowledge bases
- [Tools](../tools/index.md) — Learn about tool types
- [Evaluation](../evaluation/index.md) — Test knowledge base accuracy
