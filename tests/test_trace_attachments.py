"""Phase 5 tests — ``trace_attachments`` table + helpers.

Mock-free: every assertion runs against a real on-disk SQLite trace store
populated by the real Pillow / pymupdf code paths. No mocked DB, no mocked
HTTP — the REST endpoint is exercised separately in ``test_ui_routes.py``
where applicable.

Spec test #10 — Trace captures multimodal inputs.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image as PILImage

from fastaiagent import PDF, Image
from fastaiagent.trace.attachments import (
    get_attachment,
    list_attachments_for_span,
    save_attachment,
    save_parts_for_span,
)
from fastaiagent.trace.storage import TraceStore

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"


def _store(tmp_path: Path) -> TraceStore:
    return TraceStore(db_path=str(tmp_path / "trace.db"))


def test_save_image_persists_metadata_and_thumbnail(tmp_path: Path) -> None:
    store = _store(tmp_path)
    img = Image.from_file(FIXTURES / "cat.jpg")

    record = save_attachment(
        db=store._db,
        trace_id="t-1",
        span_id="s-1",
        data=img.data,
        media_type=img.media_type,
    )

    assert record.media_type == "image/jpeg"
    assert record.size_bytes == img.size_bytes()
    # Thumbnail must be a valid JPEG decodable by Pillow.
    assert record.thumbnail is not None
    with PILImage.open(io.BytesIO(record.thumbnail)) as thumb:
        thumb.load()
        assert max(thumb.size) <= 256
    assert record.metadata["width"] == 200
    assert record.metadata["height"] == 200


def test_save_image_does_not_store_full_data_by_default(tmp_path: Path) -> None:
    store = _store(tmp_path)
    img = Image.from_file(FIXTURES / "cat.jpg")
    record = save_attachment(
        db=store._db,
        trace_id="t-1",
        span_id="s-1",
        data=img.data,
        media_type=img.media_type,
    )
    assert record.full_data is None  # trace_full_images=False by default


def test_save_pdf_renders_first_page_thumbnail(tmp_path: Path) -> None:
    store = _store(tmp_path)
    pdf = PDF.from_file(FIXTURES / "contract.pdf")

    record = save_attachment(
        db=store._db,
        trace_id="t-2",
        span_id="s-2",
        data=pdf.data,
        media_type="application/pdf",
    )

    assert record.media_type == "application/pdf"
    assert record.thumbnail is not None
    with PILImage.open(io.BytesIO(record.thumbnail)) as thumb:
        thumb.load()
        assert max(thumb.size) <= 256
    assert record.metadata["page_count"] == 2


def test_get_attachment_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    img = Image.from_file(FIXTURES / "cat.jpg")
    saved = save_attachment(
        db=store._db,
        trace_id="t-1",
        span_id="s-1",
        data=img.data,
        media_type=img.media_type,
    )

    fetched = get_attachment(db=store._db, attachment_id=saved.attachment_id)
    assert fetched is not None
    assert fetched.attachment_id == saved.attachment_id
    assert fetched.size_bytes == img.size_bytes()
    assert fetched.thumbnail == saved.thumbnail


def test_get_attachment_unknown_id_returns_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert get_attachment(db=store._db, attachment_id="nope") is None


def test_list_attachments_for_span_filters_correctly(tmp_path: Path) -> None:
    store = _store(tmp_path)
    img = Image.from_file(FIXTURES / "cat.jpg")
    pdf = PDF.from_file(FIXTURES / "contract.pdf")

    save_attachment(
        db=store._db,
        trace_id="t-1",
        span_id="s-A",
        data=img.data,
        media_type=img.media_type,
    )
    save_attachment(
        db=store._db,
        trace_id="t-1",
        span_id="s-B",
        data=pdf.data,
        media_type="application/pdf",
    )

    a = list_attachments_for_span(db=store._db, trace_id="t-1", span_id="s-A")
    b = list_attachments_for_span(db=store._db, trace_id="t-1", span_id="s-B")
    assert len(a) == 1 and a[0].media_type == "image/jpeg"
    assert len(b) == 1 and b[0].media_type == "application/pdf"


def test_save_parts_for_span_walks_mixed_list(tmp_path: Path) -> None:
    store = _store(tmp_path)
    img = Image.from_file(FIXTURES / "cat.jpg")
    pdf = PDF.from_file(FIXTURES / "contract.pdf")

    saved = save_parts_for_span(
        db=store._db,
        trace_id="t-3",
        span_id="s-3",
        parts=["caption", img, "between", pdf, "trailing"],
        role="input",
    )
    assert len(saved) == 2
    assert {r.media_type for r in saved} == {"image/jpeg", "application/pdf"}
    # Roles + part indices preserved on metadata so the UI can sort and label.
    assert all(r.metadata.get("role") == "input" for r in saved)
    indices = sorted(r.metadata["part_index"] for r in saved)
    assert indices == [1, 3]


def test_full_data_persists_when_trace_full_images_is_true(
    tmp_path: Path, monkeypatch: Any  # noqa: F821
) -> None:
    """When ``trace_full_images=True`` is configured at SDK boot, originals
    are stored alongside thumbnails. We exercise the config flag without
    monkeypatching internals — just patch the env and reset the singleton."""
    import os

    from fastaiagent._internal.config import reset_config

    os.environ["FASTAIAGENT_TRACE_FULL_IMAGES"] = "1"
    # Re-read config — but the existing SDKConfig.from_env doesn't map this
    # env yet; in practice users set ``fa.config.trace_full_images = True``.
    reset_config()
    from fastaiagent._internal.config import get_config

    cfg = get_config()
    cfg.trace_full_images = True

    store = _store(tmp_path)
    img = Image.from_file(FIXTURES / "cat.jpg")
    record = save_attachment(
        db=store._db,
        trace_id="t-4",
        span_id="s-4",
        data=img.data,
        media_type=img.media_type,
    )
    assert record.full_data == img.data

    cfg.trace_full_images = False
    os.environ.pop("FASTAIAGENT_TRACE_FULL_IMAGES", None)
    reset_config()
