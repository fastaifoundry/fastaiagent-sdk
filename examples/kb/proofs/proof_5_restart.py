"""Proof for "5 · Between one process and the next" on docs/knowledge-base/kb-boundaries.md.

Offline, with the local FastEmbed model. No API key.

The SQLite metadata store is the source of truth. A new process rebuilds the
vector and keyword indexes from it, embeds nothing but a one-word probe, and
sees a delete that another process made. persist=False leaves nothing on disk.
"""

import subprocess
import sys
from pathlib import Path

import _common
from _common import CountingEmbedder, embedder, heading

from fastaiagent.kb import LocalKB

HERE = Path(__file__).resolve().parent
KB_DIR = str(_common.KB_DIR)
DOCS = {
    "refunds.md": "Items can be sent back within 30 days of purchase for a full refund.",
    "shipping.md": "Domestic orders ship in 3-5 business days; express delivery costs extra.",
    "errors.md": "Error code ERR-4012 means the payment gateway timed out. Retry after 30 seconds.",
}
READER = f"""
import sys; sys.path.insert(0, {str(HERE)!r})
import _common
from _common import CountingEmbedder, embedder
from fastaiagent.kb import LocalKB
emb = CountingEmbedder(embedder())
kb = LocalKB(name="policies", path={KB_DIR!r}, embedder=emb)
on_open = emb.texts
top = kb.search("can I get my money back?", top_k=1)[0]
print(f"chunks={{kb.status()['chunk_count']}} embedded_on_open={{on_open}} (the probe) "
      f"embedded_by_search={{emb.texts - on_open}} top=[{{top.score:.3f}}] {{top.chunk.content[:38]}}…")
if "--delete" in sys.argv:
    print("deleted", kb.delete_by_source("errors.md"), "chunk(s) from errors.md")
kb.close()
"""


def reader(*args: str) -> str:
    run = subprocess.run([sys.executable, "-c", READER, *args], capture_output=True, text=True,
                         cwd=str(HERE), check=True)
    return run.stdout.strip()


heading("process A writes")
counted = CountingEmbedder(embedder())
kb = LocalKB(name="policies", path=KB_DIR, embedder=counted)
from fastaiagent.kb.document import Document  # noqa: E402

kb.add_documents([Document(content=text, source=src) for src, text in DOCS.items()])
print(f"chunks={kb.status()['chunk_count']} embedded_texts={counted.texts} "
      f"files={sorted(p.name for p in (_common.KB_DIR / 'policies').iterdir())}")
kb.close()

heading("process B reads, then deletes a source")
print(reader("--delete"))
heading("process C reads what B left")
print(reader())

heading("persist=False")
scratch = LocalKB(name="scratch", path=KB_DIR, embedder=embedder(), persist=False)
scratch.add("temporary content")
print("on disk:", sorted(p.name for p in _common.KB_DIR.iterdir()))
