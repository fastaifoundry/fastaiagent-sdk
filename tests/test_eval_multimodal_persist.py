"""An image or PDF in an eval case is stored as a descriptor, not its bytes (1.89.0).

``persist_local`` wrote each case with ``json.dumps(default=str)``, which turned
an ``Image`` into its ``repr``: ``Image(data=b'\\x89PNG…')``, raw bytes and all,
into ``eval_cases.input``. Megabytes per case, unreadable in the UI, and the
image's bytes treated as text by every reader. Now the case holds
``{"type": "image", "media_type", "size_bytes", "sha256", "media_idx", …}`` and
``eval_case_media`` holds one row per item, with a thumbnail when Pillow is
installed. Real ``evaluate()``, real SQLite; no mocks.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import zlib
from pathlib import Path

import pytest

from fastaiagent.eval import evaluate
from fastaiagent.multimodal.image import Image
from fastaiagent.multimodal.pdf import PDF


def _png(width: int = 8, height: int = 8) -> bytes:
    """A real, valid RGB PNG built with the standard library."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


PDF_BYTES = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"


def test_media_is_a_descriptor_with_a_media_row(isolated_local_db: Path) -> None:
    png = _png()
    image = Image.from_bytes(png, "image/png")
    pdf = PDF.from_bytes(PDF_BYTES)
    results = evaluate(
        lambda q: "a red square",
        [{"input": ["What is this?", image, pdf], "expected": "a red square"}],
        scorers=["exact_match"],
        run_name="mm",
    )
    with sqlite3.connect(isolated_local_db) as conn:
        conn.row_factory = sqlite3.Row
        case = conn.execute(
            "SELECT case_id, input FROM eval_cases WHERE run_id = ?", (results.run_id,)
        ).fetchone()
        media = conn.execute(
            "SELECT * FROM eval_case_media WHERE case_id = ? ORDER BY idx", (case["case_id"],)
        ).fetchall()

    stored = case["input"]
    assert "data=b'" not in stored and "\\x89PNG" not in stored, "raw bytes reached local.db"
    text, img_desc, pdf_desc = json.loads(stored)
    assert text == "What is this?"
    assert img_desc == {
        "type": "image",
        "media_type": "image/png",
        "size_bytes": len(png),
        "sha256": hashlib.sha256(png).hexdigest(),
        "source_url": None,
        "detail": "auto",
        "media_idx": 0,
    }
    assert pdf_desc["type"] == "pdf" and pdf_desc["media_idx"] == 1
    assert pdf_desc["size_bytes"] == len(PDF_BYTES)

    assert [m["idx"] for m in media] == [0, 1]
    assert media[0]["media_type"] == "image/png" and media[0]["size_bytes"] == len(png)
    assert media[0]["sha256"] == hashlib.sha256(png).hexdigest()
    assert media[1]["media_type"] == "application/pdf"


def test_image_thumbnail_when_pillow_is_installed(isolated_local_db: Path) -> None:
    pytest.importorskip("PIL")
    results = evaluate(
        lambda q: "x",
        [{"input": Image.from_bytes(_png(64, 64), "image/png"), "expected": "x"}],
        scorers=["exact_match"],
    )
    with sqlite3.connect(isolated_local_db) as conn:
        thumb = conn.execute(
            "SELECT m.thumbnail FROM eval_case_media m JOIN eval_cases c ON c.case_id = m.case_id "
            "WHERE c.run_id = ?",
            (results.run_id,),
        ).fetchone()[0]
    assert thumb and thumb[:2] == b"\xff\xd8", "a JPEG thumbnail"


def test_raw_bytes_are_a_size_not_text(isolated_local_db: Path) -> None:
    results = evaluate(
        lambda q: "ok", [{"input": b"\x00\x01binary", "expected": "ok"}], scorers=["exact_match"]
    )
    with sqlite3.connect(isolated_local_db) as conn:
        stored = conn.execute(
            "SELECT input FROM eval_cases WHERE run_id = ?", (results.run_id,)
        ).fetchone()[0]
    assert json.loads(stored) == {"type": "bytes", "size_bytes": 8}


def test_plain_text_cases_are_stored_exactly_as_before(isolated_local_db: Path) -> None:
    results = evaluate(lambda q: q, [{"input": "hi", "expected": "hi"}], scorers=["exact_match"])
    with sqlite3.connect(isolated_local_db) as conn:
        row = conn.execute(
            "SELECT input, expected_output, actual_output FROM eval_cases WHERE run_id = ?",
            (results.run_id,),
        ).fetchone()
        n_media = conn.execute("SELECT COUNT(*) FROM eval_case_media").fetchone()[0]
    assert row == ('"hi"', '"hi"', '"hi"')
    assert n_media == 0
