"""Persistence helpers for ``trace_attachments`` — multimodal span artifacts.

The trace store keeps span ``attributes`` as JSON blobs, which is fine for
short text but pathological for image/PDF bytes. This module persists media
into a sibling SQLite table:

* always: ``thumbnail`` (256 px JPEG, ~30 KB) for inline UI rendering
* optionally: ``full_data`` (original bytes) when
  ``fa.config.trace_full_images`` is true and the user wants Replay to fork
  with the exact original payload

Each saved attachment gets a UUID; spans reference the IDs via
``fastaiagent.input.attachment_ids`` / ``fastaiagent.output.attachment_ids``
attributes so the UI can fetch them on demand.
"""

from __future__ import annotations

import io
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper

_THUMBNAIL_MAX_PX: int = 256
_THUMBNAIL_QUALITY: int = 75
_THUMBNAIL_MEDIA_TYPE: str = "image/jpeg"


@dataclass
class AttachmentRecord:
    """A row from the ``trace_attachments`` table."""

    attachment_id: str
    trace_id: str
    span_id: str
    media_type: str
    size_bytes: int
    thumbnail: bytes | None
    full_data: bytes | None
    metadata: dict[str, Any]
    created_at: str


def _build_thumbnail(data: bytes, media_type: str) -> tuple[bytes | None, dict[str, Any]]:
    """Return (thumbnail JPEG bytes, metadata dict) for an image payload.

    Returns ``(None, {})`` for unsupported media types — the table accepts
    NULL thumbnails so PDFs without page renders still get a row.
    """
    from PIL import Image as PILImage

    if not media_type.startswith("image/"):
        return None, {}
    try:
        with PILImage.open(io.BytesIO(data)) as pil:
            pil.load()
            width, height = pil.size
            thumb = pil.copy()
            thumb.thumbnail((_THUMBNAIL_MAX_PX, _THUMBNAIL_MAX_PX), PILImage.Resampling.LANCZOS)
            if thumb.mode != "RGB":
                thumb = thumb.convert("RGB")
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=_THUMBNAIL_QUALITY)
        return buf.getvalue(), {"width": width, "height": height}
    except Exception:
        return None, {}


def _build_pdf_thumbnail(data: bytes) -> tuple[bytes | None, dict[str, Any]]:
    """Render the first page of a PDF as a JPEG thumbnail."""
    try:
        import pymupdf

        with pymupdf.open(stream=data, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
            if doc.page_count == 0:
                return None, {}
            page = doc[0]
            pix = page.get_pixmap(dpi=72)
            png = pix.tobytes("png")
            page_count = doc.page_count
        return _build_thumbnail(png, "image/png")[0], {"page_count": page_count}
    except Exception:
        return None, {}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def save_attachment(
    *,
    db: SQLiteHelper,
    trace_id: str,
    span_id: str,
    data: bytes,
    media_type: str,
    extra_metadata: dict[str, Any] | None = None,
) -> AttachmentRecord:
    """Persist a single attachment and return the saved record.

    For ``image/*`` payloads, builds a JPEG thumbnail. For
    ``application/pdf``, renders page 1 as the thumbnail. Everything else
    is stored without a thumbnail (UI shows a generic file badge).

    ``full_data`` is persisted only when ``config.trace_full_images`` is
    truthy — by default we avoid storing the originals to keep the trace
    DB small.
    """
    cfg = get_config()
    if media_type == "application/pdf":
        thumb, pdf_meta = _build_pdf_thumbnail(data)
        meta: dict[str, Any] = dict(pdf_meta)
    else:
        thumb, img_meta = _build_thumbnail(data, media_type)
        meta = dict(img_meta)
    if extra_metadata:
        meta.update(extra_metadata)

    record = AttachmentRecord(
        attachment_id=str(uuid.uuid4()),
        trace_id=trace_id,
        span_id=span_id,
        media_type=media_type,
        size_bytes=len(data),
        thumbnail=thumb,
        full_data=data if cfg.trace_full_images else None,
        metadata=meta,
        created_at=_now_iso(),
    )
    from fastaiagent._internal.project import safe_get_project_id

    db.execute(
        """INSERT OR REPLACE INTO trace_attachments
        (attachment_id, trace_id, span_id, media_type, size_bytes,
         thumbnail, full_data, metadata_json, created_at, project_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            record.attachment_id,
            record.trace_id,
            record.span_id,
            record.media_type,
            record.size_bytes,
            record.thumbnail,
            record.full_data,
            json.dumps(meta),
            record.created_at,
            safe_get_project_id(),
        ),
    )
    return record


def save_parts_for_span(
    *,
    db: SQLiteHelper,
    trace_id: str,
    span_id: str,
    parts: list[Any],
    role: str = "input",
) -> list[AttachmentRecord]:
    """Walk a list of ``ContentPart`` and persist each ``Image``/``PDF``.

    Returns the list of records (in the same order as the parts that were
    media). ``role`` is stored in the metadata so the UI can label each
    attachment as input vs output.
    """
    from fastaiagent.multimodal.image import Image as MMImage
    from fastaiagent.multimodal.pdf import PDF as MMPDF

    saved: list[AttachmentRecord] = []
    for index, part in enumerate(parts):
        if isinstance(part, MMImage):
            saved.append(
                save_attachment(
                    db=db,
                    trace_id=trace_id,
                    span_id=span_id,
                    data=part.data,
                    media_type=part.media_type,
                    extra_metadata={"role": role, "part_index": index},
                )
            )
        elif isinstance(part, MMPDF):
            saved.append(
                save_attachment(
                    db=db,
                    trace_id=trace_id,
                    span_id=span_id,
                    data=part.data,
                    media_type="application/pdf",
                    extra_metadata={"role": role, "part_index": index},
                )
            )
    return saved


def get_attachment(
    *, db: SQLiteHelper, attachment_id: str
) -> AttachmentRecord | None:
    """Fetch a single attachment row. Returns ``None`` when absent."""
    row = db.fetchone(
        """SELECT attachment_id, trace_id, span_id, media_type, size_bytes,
                  thumbnail, full_data, metadata_json, created_at
           FROM trace_attachments WHERE attachment_id = ?""",
        (attachment_id,),
    )
    if not row:
        return None
    return AttachmentRecord(
        attachment_id=row["attachment_id"],
        trace_id=row["trace_id"],
        span_id=row["span_id"],
        media_type=row["media_type"],
        size_bytes=int(row["size_bytes"]),
        thumbnail=row["thumbnail"],
        full_data=row["full_data"],
        metadata=json.loads(row["metadata_json"] or "{}"),
        created_at=row["created_at"],
    )


def list_attachments_for_span(
    *, db: SQLiteHelper, trace_id: str, span_id: str
) -> list[AttachmentRecord]:
    """All attachments for a span, ordered by creation."""
    rows = db.fetchall(
        """SELECT attachment_id, trace_id, span_id, media_type, size_bytes,
                  thumbnail, full_data, metadata_json, created_at
           FROM trace_attachments
           WHERE trace_id = ? AND span_id = ?
           ORDER BY created_at""",
        (trace_id, span_id),
    )
    return [
        AttachmentRecord(
            attachment_id=r["attachment_id"],
            trace_id=r["trace_id"],
            span_id=r["span_id"],
            media_type=r["media_type"],
            size_bytes=int(r["size_bytes"]),
            thumbnail=r["thumbnail"],
            full_data=r["full_data"],
            metadata=json.loads(r["metadata_json"] or "{}"),
            created_at=r["created_at"],
        )
        for r in rows
    ]
