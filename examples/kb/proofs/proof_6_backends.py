"""Proof for "6 · Between a laptop and a fleet" on docs/knowledge-base/kb-boundaries.md.

Offline: FAISS, Chroma (in-process) and Qdrant (in-memory) — the real libraries,
no servers. The local FastEmbed model for the lifecycle half.

One VectorStore contract: the same three reference vectors score the same on
every backend. One LocalKB: the same lifecycle code runs over each of them.
"""

import uuid

import _common
from _common import embedder, heading

from fastaiagent.kb import LocalKB
from fastaiagent.kb.backends.chroma import ChromaVectorStore
from fastaiagent.kb.backends.faiss import FaissVectorStore
from fastaiagent.kb.backends.qdrant import QdrantVectorStore
from fastaiagent.kb.chunking import Chunk
from fastaiagent.kb.document import Document

BACKENDS = {
    "faiss": lambda dim: FaissVectorStore(dimension=dim),
    "chroma": lambda dim: ChromaVectorStore(collection=f"c{uuid.uuid4().hex[:8]}", dimension=dim),
    "qdrant": lambda dim: QdrantVectorStore(collection=f"q{uuid.uuid4().hex[:8]}", dimension=dim,
                                             location=":memory:"),
}

heading("the score contract: one stored vector, three queries")
E1 = [1.0, 0.0, 0.0, 0.0]
QUERIES = {"same": E1, "orthogonal": [0.0, 1.0, 0.0, 0.0], "opposite": [-1.0, 0.0, 0.0, 0.0],
           "cos=0.6": [0.6, 0.8, 0.0, 0.0]}
for name, make in BACKENDS.items():
    store = make(4)
    store.add([Chunk(content="e1")], [E1])
    scores = {q: round(store.search(v, 1)[0][1], 3) for q, v in QUERIES.items()}
    print(f"  {name:<7} {scores}")



class PythonFloats:
    """FastEmbed returns numpy float32 values inside Python lists; this hands on plain floats."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self.inner.embed(texts)]


def lifecycle(name: str, emb) -> str:
    kb = LocalKB(name=f"kb-{name}", path=str(_common.KB_DIR), embedder=emb,
                 vector_store=BACKENDS[name](384), persist=False)
    kb.add_documents([
        Document(content="Items can be sent back within 30 days of purchase for a full refund.",
                 source="refunds.md"),
        Document(content="Error code ERR-4012 means the payment gateway timed out.", source="errors.md"),
        Document(content="Support hours are Monday to Friday, 9am to 5pm EST.", source="hours.md"),
    ])
    before = kb.search("can I get my money back?", top_k=1)[0]
    removed = kb.delete_by_source("refunds.md")
    after = kb.search("can I get my money back?", top_k=1)[0]
    return (f"{kb.status()['vector_backend']:<17} top before delete: {before.chunk.metadata['source']} "
            f"· removed {removed} · top after: {after.chunk.metadata['source']}")


heading("the same LocalKB lifecycle over each backend, FastEmbed vectors as they come")
emb = embedder()
for name in BACKENDS:
    try:
        print(f"  {name:<7} {lifecycle(name, emb)}")
    except ValueError as e:
        print(f"  {name:<7} ValueError from the backend: {str(e)[:96]}…")

heading("the same lifecycle, vectors converted to plain Python floats")
for name in BACKENDS:
    print(f"  {name:<7} {lifecycle(name, PythonFloats(emb))}")
