"""Tests for ``fastaiagent.multimodal.Image``.

Real Pillow throughout — no mocks. Fixtures are generated on-disk by
``tests/fixtures/multimodal/_make_fixtures.py`` and committed to the repo;
re-running that script regenerates them idempotently.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image as PILImage

from fastaiagent import Image
from fastaiagent._internal.errors import MultimodalError, UnsupportedFormatError

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"


def test_from_file_jpeg_sniffs_media_type() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    assert img.media_type == "image/jpeg"
    assert img.size_bytes() > 0
    assert img.dimensions() == (200, 200)


def test_from_file_png_sniffs_media_type() -> None:
    img = Image.from_file(FIXTURES / "receipt.png")
    assert img.media_type == "image/png"
    assert img.dimensions() == (600, 800)


def test_from_bytes_explicit_media_type() -> None:
    pil = PILImage.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    img = Image.from_bytes(buf.getvalue(), media_type="image/png")
    assert img.media_type == "image/png"
    assert img.dimensions() == (10, 10)


def test_to_base64_round_trip() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    encoded = img.to_base64()
    assert base64.b64decode(encoded) == img.data


def test_to_dict_from_dict_round_trip() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg", detail="high")
    d = img.to_dict()
    assert d["type"] == "image"
    assert d["media_type"] == "image/jpeg"
    assert d["detail"] == "high"
    img2 = Image.from_dict(d)
    assert img2.data == img.data
    assert img2.media_type == img.media_type
    assert img2.detail == img.detail


def test_unsupported_media_type_raises() -> None:
    with pytest.raises(UnsupportedFormatError):
        Image.from_bytes(b"\x00\x01", media_type="image/bmp")


def test_invalid_detail_raises() -> None:
    with pytest.raises(MultimodalError):
        Image.from_bytes(b"\x00", media_type="image/png", detail="ultra")


def test_empty_data_raises() -> None:
    with pytest.raises(MultimodalError):
        Image.from_bytes(b"", media_type="image/png")


def test_from_url_rejects_file_scheme() -> None:
    with pytest.raises(UnsupportedFormatError):
        Image.from_url("file:///etc/passwd")


def test_from_url_rejects_data_scheme() -> None:
    with pytest.raises(UnsupportedFormatError):
        Image.from_url("data:image/png;base64,iVBORw0KGgo=")


def test_dataclass_equality_by_value() -> None:
    a = Image.from_file(FIXTURES / "cat.jpg")
    b = Image.from_file(FIXTURES / "cat.jpg")
    assert a == b
