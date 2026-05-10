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


def test_from_url_rejects_loopback_literal() -> None:
    """Regression for security_review_1.md C4: SSRF to 127.0.0.1."""
    with pytest.raises(MultimodalError, match="non-public|public address"):
        Image.from_url("http://127.0.0.1:7842/api/auth/status")


def test_from_url_rejects_private_rfc1918_literal() -> None:
    """Regression for security_review_1.md C4: SSRF to RFC 1918."""
    with pytest.raises(MultimodalError, match="non-public|public address"):
        Image.from_url("http://192.168.1.1/scan.png")


def test_from_url_rejects_link_local_metadata_literal() -> None:
    """Regression for security_review_1.md C4: cloud-metadata SSRF."""
    with pytest.raises(MultimodalError, match="non-public|public address"):
        Image.from_url("http://169.254.169.254/latest/meta-data/")


def test_from_url_allows_private_when_env_set(monkeypatch) -> None:
    """The opt-out env var lets intranet users keep working.

    We don't actually serve anything on 127.0.0.1 in the test, so the
    request will fail with a connection / HTTP error rather than the SSRF
    guard — which is the point: the guard no longer trips.
    """
    from fastaiagent.multimodal._http import ALLOW_PRIVATE_NETWORKS_ENV

    monkeypatch.setenv(ALLOW_PRIVATE_NETWORKS_ENV, "1")
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - any non-SSRF error is fine
        Image.from_url("http://127.0.0.1:1/does-not-exist.png")
    msg = str(excinfo.value).lower()
    assert "non-public" not in msg and "public address" not in msg


# ---------------------------------------------------------------------------
# v1.11.0 M9 — verify=True against real TLS endpoint
# ---------------------------------------------------------------------------


def test_from_url_real_public_https_image() -> None:
    """Live end-to-end fetch of a public HTTPS image — exercises the
    explicit ``verify=True`` path in ``safe_http_fetch`` against a real
    TLS endpoint, not a mock.

    Uses ``httpbin.org/image/png`` (stable public test endpoint).
    Skips if the host isn't reachable so the suite stays usable offline.
    """
    import socket as _socket

    try:
        _socket.gethostbyname("httpbin.org")
    except OSError:
        pytest.skip("httpbin.org not reachable (offline?)")

    img = Image.from_url("https://httpbin.org/image/png")
    assert img.media_type == "image/png"
    assert img.size_bytes() > 0
    width, height = img.dimensions()
    assert width > 0 and height > 0


# ---------------------------------------------------------------------------
# security_review_1.md H9 — PIL decompression-bomb cap
# ---------------------------------------------------------------------------


def test_pil_pixel_cap_is_lowered() -> None:
    """A small attacker PNG that decodes to >64 megapixels must raise.

    ``Pillow.Image.MAX_IMAGE_PIXELS`` defaults to ~89 MP and only emits a
    warning by default. We lower it and turn it into an error so a
    1-MB attacker PNG cannot OOM the worker.
    """
    from PIL import Image as PILImage

    # Build a real 9000x9000 PNG (81 MP) — slightly above our 64 MP cap.
    big = PILImage.new("RGB", (9000, 9000), color="red")
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    buf.seek(0)

    # Either: raises during decode (DecompressionBombError) OR returns
    # ``None`` from sniff. Both outcomes mean the cap is active. We accept
    # either to stay decoupled from Pillow's internal exception class
    # changes across versions.
    img = Image.from_bytes(buf.getvalue(), media_type="image/png")
    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        img.dimensions()
    msg = str(excinfo.value).lower()
    assert "pixel" in msg or "decompression" in msg or "bomb" in msg


def test_dataclass_equality_by_value() -> None:
    a = Image.from_file(FIXTURES / "cat.jpg")
    b = Image.from_file(FIXTURES / "cat.jpg")
    assert a == b
