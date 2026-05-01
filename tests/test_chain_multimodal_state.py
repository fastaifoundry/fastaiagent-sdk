"""Phase 4 tests — Chain state with Image/PDF round-trips through real SQLiteCheckpointer.

Mock-free: every test exercises the real ``ChainState`` snapshot/restore
path and the real on-disk SQLite checkpoint store. No subprocess (the
checkpointer already round-trips through bytes-on-disk; the bytes-equality
assertion proves persistence).

Spec test #7 — Chain multimodal state pass-through.
Spec test #8 — Checkpoint serialization survives kill/resume.
"""

from __future__ import annotations

from pathlib import Path

from fastaiagent import PDF, Image, SQLiteCheckpointer
from fastaiagent.chain.checkpoint import Checkpoint
from fastaiagent.chain.state import (
    ChainState,
    _hydrate_from_checkpoint,
    _serialize_for_checkpoint,
)

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"


# --- ChainState snapshot/restore (pure unit) ---


def test_snapshot_serializes_image_to_dict() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    state = ChainState({"caption": "test", "photo": img})
    snap = state.snapshot()
    assert snap["caption"] == "test"
    assert isinstance(snap["photo"], dict)
    assert snap["photo"]["type"] == "image"
    assert snap["photo"]["media_type"] == "image/jpeg"


def test_snapshot_serializes_pdf_to_dict() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    state = ChainState({"doc": pdf})
    snap = state.snapshot()
    assert snap["doc"]["type"] == "pdf"
    assert "data_base64" in snap["doc"]


def test_snapshot_handles_nested_lists_and_dicts() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    state = ChainState(
        {
            "attachments": [img, pdf],
            "meta": {"first_image": img, "tag": "claim"},
        }
    )
    snap = state.snapshot()
    assert snap["attachments"][0]["type"] == "image"
    assert snap["attachments"][1]["type"] == "pdf"
    assert snap["meta"]["first_image"]["type"] == "image"
    assert snap["meta"]["tag"] == "claim"


def test_from_snapshot_round_trips_image_bytes() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    state = ChainState({"photo": img})
    snap = state.snapshot()
    restored = ChainState.from_snapshot(snap)
    assert isinstance(restored["photo"], Image)
    assert restored["photo"].data == img.data
    assert restored["photo"].media_type == img.media_type


def test_from_snapshot_round_trips_pdf_bytes() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    state = ChainState({"doc": pdf})
    snap = state.snapshot()
    restored = ChainState.from_snapshot(snap)
    assert isinstance(restored["doc"], PDF)
    assert restored["doc"].data == pdf.data


def test_init_with_dict_marker_hydrates_back_to_objects() -> None:
    """Resume path: ``Chain.aresume`` passes a raw dict back through
    ``ChainState(initial=...)``. Hydration must happen there too."""
    img = Image.from_file(FIXTURES / "cat.jpg")
    snap = ChainState({"photo": img}).snapshot()
    restored = ChainState(snap)
    assert isinstance(restored["photo"], Image)
    assert restored["photo"].data == img.data


def test_helpers_exposed_for_external_use() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    serialized = _serialize_for_checkpoint({"a": [img]})
    assert serialized["a"][0]["type"] == "image"
    rehydrated = _hydrate_from_checkpoint(serialized)
    assert isinstance(rehydrated["a"][0], Image)


# --- Real SQLiteCheckpointer round-trip (Test #8) ---


def test_checkpoint_persists_image_bytes_to_sqlite_and_back(tmp_path: Path) -> None:
    """A ``Checkpoint`` whose ``state_snapshot`` carries an ``Image`` survives a
    write/read cycle through the on-disk SQLite store byte-for-byte."""
    db_path = tmp_path / "ck.db"
    store = SQLiteCheckpointer(db_path=str(db_path))
    store.setup()

    img = Image.from_file(FIXTURES / "cat.jpg")
    state = ChainState({"photo": img, "step": 1})

    cp = Checkpoint(
        chain_name="test_chain",
        execution_id="exec-mm-1",
        node_id="node-a",
        status="completed",
        state_snapshot=state.snapshot(),
        node_input={"trigger": "start"},
        node_output={"ok": True},
    )
    store.put(cp)

    loaded = store.get_last("exec-mm-1")
    assert loaded is not None
    assert loaded.state_snapshot["step"] == 1
    rebuilt = ChainState(loaded.state_snapshot)
    assert isinstance(rebuilt["photo"], Image)
    assert rebuilt["photo"].data == img.data
    assert rebuilt["photo"].media_type == "image/jpeg"


def test_checkpoint_persists_pdf_bytes_to_sqlite_and_back(tmp_path: Path) -> None:
    db_path = tmp_path / "ck.db"
    store = SQLiteCheckpointer(db_path=str(db_path))
    store.setup()

    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    state = ChainState({"contract": pdf})

    cp = Checkpoint(
        chain_name="test_chain",
        execution_id="exec-mm-2",
        node_id="node-b",
        status="completed",
        state_snapshot=state.snapshot(),
    )
    store.put(cp)

    loaded = store.get_last("exec-mm-2")
    assert loaded is not None
    rebuilt = ChainState(loaded.state_snapshot)
    assert isinstance(rebuilt["contract"], PDF)
    assert rebuilt["contract"].data == pdf.data
    assert rebuilt["contract"].page_count() == 2


def test_checkpoint_with_mixed_attachments_survives_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "ck.db"
    store = SQLiteCheckpointer(db_path=str(db_path))
    store.setup()

    img = Image.from_file(FIXTURES / "cat.jpg")
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    state = ChainState(
        {
            "attachments": [img, pdf],
            "meta": {"customer": "ACME"},
        }
    )

    cp = Checkpoint(
        chain_name="claim_chain",
        execution_id="exec-mm-3",
        node_id="ingest",
        status="completed",
        state_snapshot=state.snapshot(),
    )
    store.put(cp)

    loaded = store.get_last("exec-mm-3")
    assert loaded is not None
    rebuilt = ChainState(loaded.state_snapshot)
    attachments = rebuilt["attachments"]
    assert isinstance(attachments[0], Image)
    assert isinstance(attachments[1], PDF)
    assert attachments[0].data == img.data
    assert attachments[1].data == pdf.data
    assert rebuilt["meta"]["customer"] == "ACME"
