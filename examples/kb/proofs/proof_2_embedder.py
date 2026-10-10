"""Proof for "2 · Between text and vector" on docs/knowledge-base/kb-boundaries.md.

Offline, with the local FastEmbed model (pip install "fastaiagent[kb]"). No API key.

The embedder is part of the index: the query is embedded the same way as the
chunks, a meaningless embedder finds the wrong chunk, a different embedder can't
open the index, and a reopened index embeds nothing but a one-word probe.
"""

import _common
from _common import CountingEmbedder, embedder, heading

from fastaiagent.kb import LocalKB
from fastaiagent.kb.embedding import SimpleEmbedder

DOCS = [
    "Items can be sent back within 30 days of purchase for a full refund.",
    "Domestic orders ship in 3-5 business days; express delivery costs extra.",
    "Error code ERR-4012 means the payment gateway timed out. Retry after 30 seconds.",
    "Support hours are Monday to Friday, 9am to 5pm EST.",
]
QUERY = "can I get my money back?"  # no word in common with the refund chunk
MISS = "how long is the reimbursement window?"

heading("two queries, two embedders")
for label, emb in (("FastEmbed (bge-small, 384-d)", embedder()), ("SimpleEmbedder (char counts)", SimpleEmbedder())):
    kb = LocalKB(name=f"probe-{label[:5].lower()}", path=str(_common.KB_DIR), embedder=emb,
                 search_type="vector", persist=False)
    for d in DOCS:
        kb.add(d)
    for q in (QUERY, MISS):
        top = kb.search(q, top_k=1)[0]
        print(f"{label:<30} {q!r:<40} → [{top.score:.3f}] {top.chunk.content[:44]}…")

heading("the embedder is part of the index")
counted = CountingEmbedder(embedder())
kb = LocalKB(name="policies", path=str(_common.KB_DIR), embedder=counted)
for d in DOCS:
    kb.add(d)
print(f"indexing {len(DOCS)} documents embedded {counted.texts} texts")
kb.close()
try:
    LocalKB(name="policies", path=str(_common.KB_DIR), embedder=SimpleEmbedder(dimensions=128))
except ValueError as e:
    print("reopen with another embedder →", e)
counted = CountingEmbedder(embedder())
kb = LocalKB(name="policies", path=str(_common.KB_DIR), embedder=counted)
print(f"reopen with the same embedder → {kb.status()['chunk_count']} chunks back, "
      f"{counted.texts} text embedded (a one-word dimension probe)")
top = kb.search(QUERY, top_k=1)[0]
print(f"one search                    → {counted.texts} texts embedded in total: the probe and the query; "
      f"[{top.score:.3f}] {top.chunk.content[:40]}…")
kb.close()
