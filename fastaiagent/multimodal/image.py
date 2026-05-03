"""``Image`` — a first-class image input for multimodal LLM calls."""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from fastaiagent._internal.errors import MultimodalError, UnsupportedFormatError

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)
_PIL_FORMAT_TO_MEDIA_TYPE: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}
_VALID_DETAIL: frozenset[str] = frozenset({"auto", "low", "high"})
_FROM_URL_TIMEOUT_SECONDS: float = 30.0
_FROM_URL_MAX_REDIRECTS: int = 5


@dataclass
class Image:
    """An image input for multimodal LLM calls.

    Construct via :py:meth:`from_file`, :py:meth:`from_url`, or
    :py:meth:`from_bytes`. Supported formats: JPEG, PNG, GIF, WebP.
    """

    data: bytes
    media_type: str
    source_url: str | None = None
    detail: str = "auto"

    def __post_init__(self) -> None:
        if self.media_type not in SUPPORTED_IMAGE_TYPES:
            raise UnsupportedFormatError(
                f"unsupported image media_type {self.media_type!r}; "
                f"supported: {sorted(SUPPORTED_IMAGE_TYPES)}"
            )
        if self.detail not in _VALID_DETAIL:
            raise MultimodalError(
                f"invalid detail {self.detail!r}; expected one of {sorted(_VALID_DETAIL)}"
            )
        if not isinstance(self.data, bytes) or len(self.data) == 0:
            raise MultimodalError("Image.data must be non-empty bytes")

    # --- constructors ---

    @classmethod
    def from_file(cls, path: str | Path, *, detail: str = "auto") -> Image:
        """Read an image from a local path. Sniffs the media_type from content."""
        p = Path(path)
        data = p.read_bytes()
        media_type = _sniff_media_type(data) or _guess_media_type_from_path(p)
        if media_type is None:
            raise UnsupportedFormatError(f"could not determine image media_type for {p}")
        return cls(data=data, media_type=media_type, detail=detail)

    @classmethod
    def from_url(cls, url: str, *, detail: str = "auto") -> Image:
        """Fetch an image from an HTTP(S) URL. Times out at 30s, max 5 redirects.

        Other schemes (``file://``, ``data:``) are rejected.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise UnsupportedFormatError(
                f"unsupported URL scheme {parsed.scheme!r}; only http(s) allowed"
            )
        with httpx.Client(
            follow_redirects=True,
            max_redirects=_FROM_URL_MAX_REDIRECTS,
            timeout=_FROM_URL_TIMEOUT_SECONDS,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
        data = resp.content
        header_ct = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        media_type = header_ct or _sniff_media_type(data)
        if media_type not in SUPPORTED_IMAGE_TYPES:
            sniffed = _sniff_media_type(data)
            if sniffed in SUPPORTED_IMAGE_TYPES:
                media_type = sniffed
        if media_type not in SUPPORTED_IMAGE_TYPES:
            raise UnsupportedFormatError(f"URL returned unsupported media_type {media_type!r}")
        return cls(data=data, media_type=media_type, source_url=url, detail=detail)

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str, *, detail: str = "auto") -> Image:
        """Construct from raw bytes and an explicit media_type."""
        return cls(data=data, media_type=media_type, detail=detail)

    # --- serialization ---

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "image",
            "data_base64": self.to_base64(),
            "media_type": self.media_type,
            "source_url": self.source_url,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Image:
        return cls(
            data=base64.b64decode(d["data_base64"]),
            media_type=d["media_type"],
            source_url=d.get("source_url"),
            detail=d.get("detail", "auto"),
        )

    # --- introspection ---

    def size_bytes(self) -> int:
        return len(self.data)

    def dimensions(self) -> tuple[int, int]:
        """Return (width, height) by decoding via Pillow. Raises on undecodable input."""
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(self.data)) as img:
            return img.size


# --- helpers ---


def _sniff_media_type(data: bytes) -> str | None:
    """Sniff an image media_type from raw bytes via Pillow's format detection."""
    try:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(data)) as img:
            fmt = img.format
    except Exception:
        logger.debug("Failed to sniff image media type from raw bytes", exc_info=True)
        return None
    if fmt is None:
        return None
    return _PIL_FORMAT_TO_MEDIA_TYPE.get(fmt.upper())


def _guess_media_type_from_path(path: Path) -> str | None:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed in SUPPORTED_IMAGE_TYPES:
        return guessed
    return None
