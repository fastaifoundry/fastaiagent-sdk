"""Real-Pillow tests for ``maybe_resize`` (no mocks).

Test #13 from the spec — auto-resize triggers above the configured size, the
output decodes successfully, and a warning is logged so the developer notices.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest
from PIL import Image as PILImage

from fastaiagent import Image
from fastaiagent.multimodal.resize import maybe_resize

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"


def _make_large_jpeg(target_bytes: int) -> Image:
    """Build an Image whose encoded JPEG byte count is at least ``target_bytes``.

    Uses random-ish noise so JPEG can't squeeze it down.
    """
    import random

    rng = random.Random(0xCAFE)
    side = 256
    while True:
        pil = PILImage.new("RGB", (side, side))
        pixels = [
            (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            for _ in range(side * side)
        ]
        pil.putdata(pixels)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95)
        if buf.tell() >= target_bytes or side >= 8192:
            return Image(data=buf.getvalue(), media_type="image/jpeg")
        side *= 2


def test_no_resize_when_under_budget() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    out = maybe_resize(img, max_mb=20.0)
    assert out is img


def test_resize_when_over_budget(caplog: pytest.LogCaptureFixture) -> None:
    big = _make_large_jpeg(target_bytes=3 * 1024 * 1024)  # ~3 MB
    cap_mb = 0.5
    with caplog.at_level(logging.WARNING):
        out = maybe_resize(big, max_mb=cap_mb)
    assert out is not big
    assert out.size_bytes() <= int(cap_mb * 1024 * 1024)
    # The result must still decode as a valid image.
    with PILImage.open(io.BytesIO(out.data)) as decoded:
        decoded.load()
    assert any("auto-resized image" in rec.message for rec in caplog.records)


def test_resize_preserves_media_type_for_jpeg() -> None:
    big = _make_large_jpeg(target_bytes=2 * 1024 * 1024)
    out = maybe_resize(big, max_mb=0.5)
    # JPEG inputs stay JPEG (we only fall back to PNG for non-JPEG sources).
    assert out.media_type == "image/jpeg"


def test_resize_preserves_source_url_and_detail() -> None:
    big = _make_large_jpeg(target_bytes=2 * 1024 * 1024)
    big = Image(
        data=big.data,
        media_type="image/jpeg",
        source_url="https://example.com/big.jpg",
        detail="high",
    )
    out = maybe_resize(big, max_mb=0.5)
    assert out.source_url == "https://example.com/big.jpg"
    assert out.detail == "high"
