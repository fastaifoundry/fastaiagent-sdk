"""Example 45: Multimodal — image flows through Chain state and survives a
checkpoint round-trip.

Demonstrates:

1. ``Image`` placed in ``ChainState`` is serialized to a JSON-safe dict on
   ``snapshot()`` and rehydrated to a real ``Image`` on ``from_snapshot()``
   — no manual conversion required.
2. A real on-disk SQLite checkpoint round-trips the bytes byte-for-byte.
3. A simulated process restart (fresh checkpointer pointing at the same
   DB file) loads the snapshot back and the rehydrated state is usable
   by the next chain node.

No LLM key is required — this exercises the SDK plumbing only. The vision
agent pass over the loaded image is left to ``examples/43_multimodal_image.py``.

Usage::

    python examples/45_multimodal_chain.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastaiagent import PDF, Image, SQLiteCheckpointer
from fastaiagent.chain.checkpoint import Checkpoint
from fastaiagent.chain.state import ChainState

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "multimodal"


def _ensure_fixtures() -> tuple[Path, Path]:
    cat = FIXTURES / "cat.jpg"
    pdf = FIXTURES / "contract.pdf"
    if not cat.exists() or not pdf.exists():
        from tests.fixtures.multimodal._make_fixtures import main as make

        make()
    return cat, pdf


def main() -> None:
    cat_path, pdf_path = _ensure_fixtures()

    # Step 1 — build a state with an Image and a PDF.
    img = Image.from_file(cat_path)
    pdf = PDF.from_file(pdf_path)
    state = ChainState(
        {
            "claim_id": "C-101",
            "photo": img,
            "policy": pdf,
            "step": "ingest",
        }
    )
    print(f"original photo:   {img.size_bytes()} bytes ({img.media_type})")
    print(f"original policy:  {pdf.size_bytes()} bytes ({pdf.page_count()} pages)")

    # Step 2 — snapshot and inspect the wire shape.
    snap = state.snapshot()
    print()
    print("snapshot keys:", sorted(snap.keys()))
    print(
        "photo  serialized as dict:",
        isinstance(snap["photo"], dict),
        snap["photo"]["type"],
    )
    print(
        "policy serialized as dict:",
        isinstance(snap["policy"], dict),
        snap["policy"]["type"],
    )

    # Step 3 — persist via real SQLite.
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "chain.db")
        store = SQLiteCheckpointer(db_path=db_path)
        store.setup()
        store.put(
            Checkpoint(
                chain_name="multimodal-example",
                execution_id="ex-45",
                node_id="ingest",
                status="completed",
                state_snapshot=snap,
                node_input={"trigger": "claim received"},
                node_output={"ok": True},
            )
        )

        # Step 4 — fresh checkpointer instance points at the same file:
        # this is what a restarted process sees.
        store_after_restart = SQLiteCheckpointer(db_path=db_path)
        loaded = store_after_restart.get_last("ex-45")
        assert loaded is not None
        rebuilt = ChainState(loaded.state_snapshot)

        print()
        print("after restart:")
        print(f"  claim_id        = {rebuilt['claim_id']}")
        print(f"  photo type      = {type(rebuilt['photo']).__name__}")
        photo = rebuilt["photo"]
        print(f"  photo bytes     = {photo.size_bytes()} ({photo.media_type})")
        print(f"  bytes identical = {rebuilt['photo'].data == img.data}")
        print(f"  policy type     = {type(rebuilt['policy']).__name__}")
        print(f"  policy pages    = {rebuilt['policy'].page_count()}")
        print(f"  bytes identical = {rebuilt['policy'].data == pdf.data}")


if __name__ == "__main__":
    main()
