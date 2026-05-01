"""``PDF`` — a first-class PDF input for multimodal LLM calls.

The SDK supports two processing modes:

* **text** — extract text via ``pymupdf`` and send as plain text. Cheap, fast,
  loses visual layout.
* **vision** — render each page as an image and send those to a vision LLM.
  More expensive, preserves layout (tables, charts, signatures).

Mode selection happens at the ``LLMClient`` boundary based on the configured
``pdf_mode`` and the model's vision capability.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from fastaiagent._internal.errors import MultimodalError, UnsupportedFormatError
from fastaiagent.multimodal.image import Image

_PDF_MEDIA_TYPE = "application/pdf"
_FROM_URL_TIMEOUT_SECONDS: float = 30.0
_FROM_URL_MAX_REDIRECTS: int = 5
_DEFAULT_RENDER_DPI: int = 150

logger = logging.getLogger(__name__)


@dataclass
class PDF:
    """A PDF input for multimodal LLM calls.

    Construct via :py:meth:`from_file`, :py:meth:`from_url`, or
    :py:meth:`from_bytes`. Use :py:meth:`extract_text` for text-mode pipelines
    and :py:meth:`to_page_images` for vision-mode pipelines.
    """

    data: bytes
    source_path: str | None = None
    source_url: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or len(self.data) == 0:
            raise MultimodalError("PDF.data must be non-empty bytes")
        if not self.data.startswith(b"%PDF-"):
            raise UnsupportedFormatError("PDF data does not start with '%PDF-' magic bytes")

    # --- constructors ---

    @classmethod
    def from_file(cls, path: str | Path) -> PDF:
        p = Path(path)
        return cls(data=p.read_bytes(), source_path=str(p))

    @classmethod
    def from_bytes(cls, data: bytes) -> PDF:
        return cls(data=data)

    @classmethod
    def from_url(cls, url: str) -> PDF:
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
        return cls(data=resp.content, source_url=url)

    # --- processing ---

    @property
    def media_type(self) -> str:
        return _PDF_MEDIA_TYPE

    def page_count(self) -> int:
        import pymupdf

        with pymupdf.open(stream=self.data, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
            return int(doc.page_count)

    def extract_text(self) -> str:
        """Extract text from all pages, joined with double newlines.

        Uses ``pymupdf`` (a.k.a. ``fitz``).
        """
        import pymupdf

        parts: list[str] = []
        with pymupdf.open(stream=self.data, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
            for page in doc:
                parts.append(page.get_text())
        return "\n\n".join(parts)

    def to_page_images(
        self,
        *,
        dpi: int = _DEFAULT_RENDER_DPI,
        max_pages: int | None = None,
    ) -> list[Image]:
        """Render each page as an :class:`Image` using ``pymupdf``.

        ``max_pages`` truncates with a warning log. ``dpi`` controls render
        resolution; 150 DPI balances clarity against payload size.
        """
        import pymupdf

        images: list[Image] = []
        with pymupdf.open(stream=self.data, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
            total = doc.page_count
            limit = total if max_pages is None else min(total, max_pages)
            if max_pages is not None and total > max_pages:
                logger.warning(
                    "PDF has %d pages; truncating to %d for vision-mode rendering",
                    total,
                    max_pages,
                )
            for i in range(limit):
                page = doc[i]
                pix = page.get_pixmap(dpi=dpi)
                png_bytes = pix.tobytes("png")
                images.append(Image.from_bytes(png_bytes, media_type="image/png"))
        return images

    # --- serialization ---

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "pdf",
            "data_base64": self.to_base64(),
            "source_path": self.source_path,
            "source_url": self.source_url,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PDF:
        return cls(
            data=base64.b64decode(d["data_base64"]),
            source_path=d.get("source_path"),
            source_url=d.get("source_url"),
        )

    def size_bytes(self) -> int:
        return len(self.data)
