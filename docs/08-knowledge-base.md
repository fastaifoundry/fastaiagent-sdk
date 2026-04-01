# Knowledge Base

LocalKB provides a file-based knowledge base with document ingestion, recursive text chunking, embedding, and cosine similarity search. It works entirely offline and can be plugged into any agent as a tool.

## Quick Start

```python
from fastaiagent.kb import LocalKB

kb = LocalKB(name="product-docs")

# Add content (text or files)
kb.add("Refund policy: Returns accepted within 30 days of purchase.")
kb.add("Shipping: 3-5 business days domestic, 7-14 days international.")
kb.add("/path/to/faq.md")

# Search
results = kb.search("How do I return an item?", top_k=3)
for r in results:
    print(f"[{r.score:.3f}] {r.chunk.content[:80]}...")
```

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

Each file is:
1. Read and extracted to text (PDF pages extracted individually)
2. Split into chunks (default 512 characters with 50-character overlap)
3. Embedded into vectors
4. Stored in memory for search

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

## Searching

```python
results = kb.search("refund policy", top_k=5)

for r in results:
    print(f"Score: {r.score:.3f}")
    print(f"Content: {r.chunk.content}")
    print(f"Source: {r.chunk.metadata.get('source', 'unknown')}")
    print()
```

**How search works:**
1. Your query is embedded using the same embedder as the documents
2. Cosine similarity is computed against every chunk's embedding
3. Top-K results are returned, sorted by score (highest first)

### Empty KB

Searching an empty KB returns an empty list — no error:

```python
kb = LocalKB(name="empty")
results = kb.search("anything")
print(len(results))  # 0
```

## Using KB as an Agent Tool

The most common pattern — give your agent access to a knowledge base:

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.kb import LocalKB

# Build the KB
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
    print(f"[{c.index}] {c.start_char}-{c.end_char}: {c.content[:40]}...")
```

## Search Result

| Field | Type | Description |
|-------|------|-------------|
| `chunk` | `Chunk` | The matched chunk |
| `score` | `float` | Cosine similarity score (0.0–1.0) |

### Chunk

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Chunk text |
| `metadata` | `dict` | Source info, custom metadata |
| `index` | `int` | Position in the original document |
| `start_char` | `int` | Start character offset |
| `end_char` | `int` | End character offset |

## KB Status

```python
status = kb.status()
print(status)
# {"name": "product-docs", "chunk_count": 15, "path": ".fastaiagent/kb"}
```

## Cosine Similarity

The search function under the hood:

```python
from fastaiagent.kb.search import cosine_similarity

score = cosine_similarity(vector_a, vector_b)
# 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
```

## Storage

KB data is stored in memory during the session. The path parameter controls where auxiliary files go:

```python
kb = LocalKB(name="docs", path=".fastaiagent/kb/")
```

Default: `.fastaiagent/kb/`

> **Note:** The current implementation keeps chunks and embeddings in memory. For large knowledge bases in production, use the platform KB which provides persistent vector storage with hybrid search, reranking, and contextual enrichment.

## CLI Commands

```bash
# Check KB status
fastaiagent kb status --name product-docs

# Add a file
fastaiagent kb add ./docs/readme.md --name product-docs
```

Example:
```
$ fastaiagent kb status --name product-docs
KB: product-docs
Chunks: 15
Path: .fastaiagent/kb

$ fastaiagent kb add ./docs/faq.md --name product-docs
Added 8 chunks from ./docs/faq.md
```

## Error Handling

```python
from fastaiagent._internal.errors import KBError

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
```

## Complete Example

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.kb import LocalKB
from fastaiagent.kb.embedding import OpenAIEmbedder

# 1. Build a KB with OpenAI embeddings
kb = LocalKB(
    name="company-docs",
    embedder=OpenAIEmbedder(),
    chunk_size=512,
)

# 2. Ingest files
kb.add("docs/refund-policy.md")
kb.add("docs/shipping-guide.md")
kb.add("docs/faq.md")
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
