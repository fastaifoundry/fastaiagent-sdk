"""Proof for "3 · Between meaning and words" on docs/knowledge-base/kb-boundaries.md.

Offline, with the local FastEmbed model. No API key.

Three queries, three matchers. Vector scores are cosines; keyword scores are
BM25; hybrid min-max normalizes both and fuses them — unless one side is empty,
in which case the other side's raw scores pass straight through.
"""

import _common
from _common import embedder, heading

from fastaiagent.kb import LocalKB

DOCS = [
    "Items can be sent back within 30 days of purchase for a full refund.",
    "Domestic orders ship in 3-5 business days; express delivery costs extra.",
    "Error code ERR-4012 means the payment gateway timed out. Retry after 30 seconds.",
    "Error code ERR-5001 means authentication failed. Check your API key.",
    "Support hours are Monday to Friday, 9am to 5pm EST.",
]
QUERIES = [
    ("an exact code",            "ERR-4012"),
    ("meaning, no shared words", "reimbursement timeframe?"),
    ("both at once",             "ERR-4012 payment failed"),
]

kbs = {}
emb = embedder()
for mode in ("vector", "keyword", "hybrid"):
    kb = LocalKB(name=f"m-{mode}", path=str(_common.KB_DIR), embedder=emb, search_type=mode,
                 persist=False)
    for d in DOCS:
        kb.add(d)
    kbs[mode] = kb

for label, q in QUERIES:
    heading(f"{label}: {q!r}")
    for mode, kb in kbs.items():
        hits = kb.search(q, top_k=2)
        shown = " | ".join(f"[{h.score:.3f}] {h.chunk.content[:34]}…" for h in hits) or "(nothing)"
        print(f"  {mode:<8} {shown}")

heading("hybrid with an empty keyword side is the vector result, unnormalized")
q = "reimbursement timeframe?"
v = [(h.chunk.content, round(h.score, 6)) for h in kbs["vector"].search(q, top_k=3)]
h = [(h.chunk.content, round(h.score, 6)) for h in kbs["hybrid"].search(q, top_k=3)]
print("  identical:", v == h)
heading("hybrid with both sides: alpha · 1.0 pins the best vector hit")
top = kbs["hybrid"].search("ERR-4012 payment failed", top_k=1)[0]
print(f"  top hybrid score {top.score:.3f} = 0.7 × 1.0 (vector) + 0.3 × 1.0 (keyword) when one chunk tops both")
