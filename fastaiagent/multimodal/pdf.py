"""``PDF`` — a first-class PDF input for multimodal LLM calls.

The SDK supports three processing modes:

* **native** — forward the raw PDF to the provider, which parses it server-side.
  Supported by Anthropic (Claude 3.5+) and OpenAI/Azure vision models
  (gpt-4o/4.1/5, o-series). No local decoding at all, so it needs no PDF engine
  and handles PDFs a local parser would choke on.
* **text** — send extracted text as a plain text block. Cheap, fast, loses
  visual layout.
* **vision** — render each page as an image and send those to a vision LLM.
  More expensive, preserves layout (tables, charts, signatures).

Mode selection happens at the ``LLMClient`` boundary based on the configured
``pdf_mode`` and the model's capabilities (``pdf_mode="auto"`` prefers native).

``text`` and ``vision`` decode locally, which needs an optional engine —
``pip install "fastaiagent[pdf]"``. To avoid it, either stay on ``native`` or
supply your own text: ``PDF.from_file(path, text=my_parser.extract(path))``.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastaiagent._internal.errors import MultimodalError, UnsupportedFormatError
from fastaiagent.multimodal import _pdf_backend
from fastaiagent.multimodal._http import safe_http_fetch
from fastaiagent.multimodal.image import Image

_PDF_MEDIA_TYPE = "application/pdf"
_FROM_URL_TIMEOUT_SECONDS: float = 30.0
_FROM_URL_MAX_REDIRECTS: int = 5
_FROM_URL_MAX_BYTES: int = 100 * 1024 * 1024  # 100 MiB
_DEFAULT_RENDER_DPI: int = 150

logger = logging.getLogger(__name__)


def _pdf_parse_error(exc: Exception) -> MultimodalError:
    """Wrap a raw engine failure in an actionable :class:`MultimodalError`.

    The underlying engine raises bare ``RuntimeError``/``ValueError`` (e.g.
    "unable to flat-compressed content") that don't tell the caller what to do.
    Point them at ``pdf_mode="native"``, which forwards the raw PDF to the
    provider and skips local rendering entirely.
    """
    return MultimodalError(
        f"failed to parse/render PDF locally ({exc}); the PDF may use "
        "unsupported or malformed compression. For OpenAI/Anthropic "
        "vision-capable models use pdf_mode='native' to let the provider parse "
        "the PDF server-side, or pass pre-extracted text via PDF(text=...)."
    )


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
    #: Pre-extracted text, supplied by the caller. When set, :py:meth:`extract_text`
    #: returns it verbatim and no local PDF engine is involved — bring whichever
    #: parser you already use (pdfplumber, pypdf, Tika, a vendor OCR API) and hand
    #: the SDK its output. Survives ``to_dict``/``from_dict``, so a checkpointed
    #: chain resumes with the text intact.
    text: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or len(self.data) == 0:
            raise MultimodalError("PDF.data must be non-empty bytes")
        if not self.data.startswith(b"%PDF-"):
            raise UnsupportedFormatError("PDF data does not start with '%PDF-' magic bytes")

    # --- constructors ---

    @classmethod
    def from_file(cls, path: str | Path, *, text: str | None = None) -> PDF:
        """Read a PDF from disk.

        Pass ``text=`` to supply your own extracted text and skip local
        decoding entirely::

            pdf = PDF.from_file("contract.pdf", text=my_parser.extract("contract.pdf"))
        """
        p = Path(path)
        return cls(data=p.read_bytes(), source_path=str(p), text=text)

    @classmethod
    def from_bytes(cls, data: bytes, *, text: str | None = None) -> PDF:
        return cls(data=data, text=text)

    @classmethod
    def from_url(cls, url: str, *, text: str | None = None) -> PDF:
        """Fetch a PDF from an HTTP(S) URL. Times out at 30s, max 5 redirects.

        Rejects non-HTTP(S) schemes and refuses any host that resolves to a
        private/loopback/link-local address (SSRF hardening). Set
        ``FASTAIAGENT_ALLOW_PRIVATE_NETWORKS=1`` to opt in for intranet use.
        Body is capped at 100 MiB.
        """
        resp = safe_http_fetch(
            url,
            timeout=_FROM_URL_TIMEOUT_SECONDS,
            max_redirects=_FROM_URL_MAX_REDIRECTS,
            max_bytes=_FROM_URL_MAX_BYTES,
        )
        return cls(data=resp.content, source_url=url, text=text)

    # --- processing ---

    @property
    def media_type(self) -> str:
        return _PDF_MEDIA_TYPE

    def page_count(self) -> int:
        """Number of pages. Needs a local PDF engine (``fastaiagent[pdf]``)."""
        if not _pdf_backend.available():
            _pdf_backend.require("PDF.page_count()")
        try:
            return _pdf_backend.page_count(self.data)
        except Exception as e:
            raise _pdf_parse_error(e) from e

    def extract_text(self) -> str:
        """Text from all pages, joined with double newlines.

        Returns :py:attr:`text` verbatim when it was supplied at construction —
        that path needs no PDF engine at all. Otherwise decodes locally, which
        requires ``pip install "fastaiagent[pdf]"``.
        """
        if self.text is not None:
            return self.text
        if not _pdf_backend.available():
            _pdf_backend.require("PDF.extract_text()")
        try:
            return _pdf_backend.extract_text(self.data)
        except Exception as e:
            raise _pdf_parse_error(e) from e

    def to_page_images(
        self,
        *,
        dpi: int = _DEFAULT_RENDER_DPI,
        max_pages: int | None = None,
    ) -> list[Image]:
        """Render each page as an :class:`Image`.

        ``max_pages`` truncates with a warning log. ``dpi`` controls render
        resolution; 150 DPI balances clarity against payload size.

        Needs a local PDF engine (``fastaiagent[pdf]``). To avoid it entirely,
        render pages with your own library and pass the resulting
        :class:`Image` parts straight into the LLM call — ``normalize_input``
        accepts them.
        """
        if not _pdf_backend.available():
            _pdf_backend.require("PDF.to_page_images()")
        try:
            total = _pdf_backend.page_count(self.data)
            if max_pages is not None and total > max_pages:
                logger.warning(
                    "PDF has %d pages; truncating to %d for vision-mode rendering",
                    total,
                    max_pages,
                )
            pages = _pdf_backend.render_pages(self.data, dpi=dpi, limit=max_pages)
        except Exception as e:
            raise _pdf_parse_error(e) from e
        return [Image.from_bytes(png, media_type="image/png") for png in pages]

    # --- serialization ---

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": "pdf",
            "data_base64": self.to_base64(),
            "source_path": self.source_path,
            "source_url": self.source_url,
        }
        # Only emitted when the caller supplied it, so payloads for the common
        # case are byte-identical to pre-1.56.0 ones.
        if self.text is not None:
            d["text"] = self.text
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PDF:
        return cls(
            data=base64.b64decode(d["data_base64"]),
            source_path=d.get("source_path"),
            source_url=d.get("source_url"),
            text=d.get("text"),  # absent in payloads written before 1.56.0
        )

    def size_bytes(self) -> int:
        return len(self.data)
