# Concepts & Mental Model

This page explains **what** a knowledge base is, **why** you'd reach for one,
and **the concept of how** retrieval actually works — chunking, embeddings, the
vector/keyword/hybrid mechanics, and how a KB becomes something an agent can
use. It's the mental model; for the full API and backend options see the
[Knowledge Base reference](index.md) and [Backends](backends.md).

## What it is

A knowledge base turns a pile of documents into something an agent can *search
by meaning*. You add text; the KB splits it into chunks, converts each chunk to
a vector, and stores it. At query time it finds the chunks most relevant to a
question and hands them back. `LocalKB` does this entirely in-process (FAISS +
BM25 + SQLite, no external infra); `PlatformKB` exposes the *same* interface
against a hosted index.

## Why it exists

An LLM only knows what's in its weights and what you put in the prompt. To
answer from *your* data — docs, policies, tickets, a codebase — you have two
honest options:

- **Put everything in the prompt** (long context). Simple, but it doesn't scale:
  a 10,000-document corpus won't fit, and stuffing the window is slow and
  expensive and dilutes attention.
- **Retrieve just the relevant pieces** (RAG). Keep the corpus in a KB, and at
  question time pull back only the handful of chunks that matter.

A KB is the second option. Use it when the corpus is **larger than the context
window**, **changes often** (you can add/update documents without retraining),
or when you want **citations** to the source chunk. Reach for long context when
everything comfortably fits; reach for fine-tuning to change *behavior/style*,
not to inject *facts* — facts belong in a KB where they're cheap to update.

## The concept of how retrieval works

Retrieval is four ideas in sequence. Understanding each is the whole mental
model.

### 1. Chunking — why not store whole documents

Embeddings summarize a *span* of text into one vector; a whole document
averages out to mush, and you'd retrieve far more text than the answer needs.
So documents are split into **chunks** (default ~512 chars, 50 overlap). The
splitter is **recursive**: it tries natural boundaries in order — paragraphs
(`\n\n`), then lines, then sentences (`. `), then words — so a chunk breaks at a
semantic seam rather than mid-sentence. The **overlap** carries a little context
across the boundary so a fact split across two chunks isn't lost.

### 2. Embedding — turning text into a comparable vector

Each chunk is run through an **embedder** (auto-selected: FastEmbed → OpenAI →
a simple fallback) into a fixed-length vector where *semantic similarity ≈
geometric closeness*. The query is embedded the **same way** at search time, so
"how long do I have to return something?" lands near a chunk about a 30-day
refund window even with no shared keywords. Two consequences worth knowing:
embeddings are **cached** (stored alongside the chunk in SQLite, so a restart
never re-embeds), and the embedder is **part of the index** — switching
embedders means the query and stored vectors no longer live in the same space,
so you re-embed the corpus.

### 3. The three matchers — vector, keyword, hybrid

A KB can score a query three ways:

- **Vector** — embed the query, find the nearest chunk vectors. FAISS uses
  **inner product** on normalized vectors (equivalent to cosine similarity).
  Great at *meaning*; weak at exact tokens (a specific SKU, an error code).
- **Keyword (BM25)** — classic term-frequency/inverse-document-frequency
  scoring. Great at *exact terms and rare words*; blind to synonyms.
- **Hybrid** (default) — run both, **min-max normalize** each score list to
  `[0,1]` (so the two scales are comparable), then combine per chunk as
  `alpha * vector + (1 - alpha) * keyword` (`alpha=0.7` favors meaning). It
  over-fetches (`top_k * 3` from each matcher) before fusing so a chunk strong
  in only one signal still surfaces. Hybrid is the default because most queries
  want *both* semantic recall and exact-term precision.

!!! info "Verified against a live run"
    One policy document chunked into **3 chunks**; a hybrid search for "how long
    do I have to return something?" ranked the refund-window chunk first with
    score **0.700** — exactly `alpha(0.7) × normalized_vector(1.0)`, the
    fusion formula in action — while an off-topic gift-card chunk scored 0.019.

### 4. The result

Each hit is a `SearchResult(chunk, score)` — the chunk's text, its `metadata`
(including `source`), and the fused relevance score. That's what you cite, rank,
or feed to the model.

## How an agent uses a KB: retrieval as a tool

This is the part most RAG explanations skip. A KB doesn't silently stuff text
into the prompt. `kb.as_tool()` wraps the KB as an ordinary
[tool](../tools/concepts.md) named `search_<kb>` (origin `kb`). The agent's
model **decides when to call it**, passes a query, and gets back the formatted
top chunks as the tool result — which then enters the conversation and informs
the answer.

The trade-off vs. injecting retrieved text up front:

- **Retrieval-as-tool** (this SDK's default) — the model retrieves *on demand*,
  can search multiple times with refined queries, and only pays for context it
  asks for. It also means retrieval shows up in the trace as a `tool.search_*`
  span you can inspect.
- **Context injection** — you retrieve yourself and prepend the chunks to the
  prompt. Fine when you always want the same context, but the model can't
  choose to look again.

!!! info "Verified against a live run"
    `kb.as_tool()` produced a tool named `search_probe` with `origin="kb"`;
    calling it for "shipping cost" returned the shipping chunk formatted as
    `[Score: 0.700] …` — the exact text the model would read as the tool result.

## The architecture underneath

`LocalKB` is three independent, swappable stores behind one interface:

- **VectorStore** (FAISS by default) — the embeddings + nearest-neighbor search.
- **KeywordStore** (BM25) — the term index.
- **MetadataStore** (SQLite) — the documents, chunks, and cached embeddings that
  make persistence and restart-without-re-embedding possible.

Because they're protocols, you can point the vector store at **Qdrant** or
**Chroma** without changing your agent code ([Backends](backends.md)), or swap
the whole thing for **`PlatformKB`** — a thin HTTP client with the identical
`search()` / `as_tool()` surface, where embedding, search, and optional
reranking run on the platform. LocalKB for zero-infra dev and self-contained
apps; PlatformKB when the corpus is managed centrally and shared across agents.

## Next steps

- [Knowledge Base reference](index.md) — the full API: ingestion, search modes, CRUD, persistence, `as_tool`
- [Backends](backends.md) — swap FAISS/BM25/SQLite for Qdrant, Chroma, or your own
- [Platform KB](platform-kb.md) — hosted retrieval with the same interface
- [Tools](../tools/concepts.md) — how `as_tool()` retrieval reaches the model
- [`examples/06_rag_agent.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/06_rag_agent.py) — a RAG agent end to end
