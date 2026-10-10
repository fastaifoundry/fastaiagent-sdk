"""Proof for "1 · Between a document and its chunks" on docs/knowledge-base/kb-boundaries.md.

Offline: the chunker alone, no embedder, no model.

The splitter cuts at the first seam that fits — paragraph, line, sentence, word —
and merges pieces up to chunk_size. It records where each chunk came from. It
does not carry text across a seam: chunk_overlap is accepted and has no effect
on the chunk text.
"""

from fastaiagent.kb.chunking import chunk_text

DOC = (
    "Refund policy. Items can be sent back within 30 days of purchase for a full refund. "
    "They must be unused and in their original packaging. Digital products are final.\n\n"
    "Shipping. Domestic orders ship in 3-5 business days. International orders take 7-14 "
    "business days. Express shipping can be chosen at checkout for an extra fee.\n\n"
    "Support. Email support@example.com or call 1-800-EXAMPLE. Hours are Monday to Friday, "
    "9am to 5pm EST."
)

for size in (512, 120):
    chunks = chunk_text(DOC, chunk_size=size, overlap=50, metadata={"source": "policy.md"})
    print(f"\nchunk_size={size}, overlap=50 → {len(chunks)} chunk(s)")
    for c in chunks:
        print(f"  #{c.index} [{c.start_char:3d}-{c.end_char:3d}] {c.content[:62]!r}…"
              if len(c.content) > 62 else f"  #{c.index} [{c.start_char:3d}-{c.end_char:3d}] {c.content!r}")
    shared = [a.end_char - b.start_char for a, b in zip(chunks, chunks[1:])]
    print("  characters shared between neighbours:", shared or "n/a (one chunk)")
    print("  metadata on every chunk:", chunks[0].metadata)

# A fact that straddles a seam is split in two.
chunks = chunk_text(DOC, chunk_size=120, overlap=50)
seam = [c for c in chunks if "30 days" in c.content]
print("\n'within 30 days … full refund' lives in chunk", [c.index for c in seam],
      "· 'unused … original packaging' lives in chunk",
      [c.index for c in chunks if "original packaging" in c.content])
