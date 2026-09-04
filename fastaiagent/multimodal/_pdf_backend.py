"""The SDK's local PDF engine — the only module that names ``pypdfium2``.

Local decoding is **optional**. Three ways to get PDF text and page images,
in the order most callers should reach for them:

1. **Don't decode at all.** ``pdf_mode="native"`` forwards the raw PDF to the
   provider, which parses it server-side. This is what ``pdf_mode="auto"``
   already picks for Claude 3.5+/4.x, GPT-4o/4.1/5/o-series, Azure, Bedrock
   Claude and Gemini — so the common path never enters this module.
2. **Bring your own parser.** ``PDF.from_file(p, text=my_parser.extract(p))``
   takes text from whatever library you already use, and needs nothing here.
3. **Install the extra.** ``pip install "fastaiagent[pdf]"``.

WHY pypdfium2
``pymupdf`` was the previous engine. It is dual-licensed AGPL-3.0 / Artifex
Commercial, and FastAIAgent policy rules out paid licences — leaving AGPL,
which enterprise licence scanners flag anywhere in a dependency tree.
``pypdfium2`` is BSD-3-Clause + Apache-2.0 bindings to PDFium, Google's PDF
engine (the one in Chrome). Output parity with pymupdf was verified
pixel-for-pixel at 72 and 150 DPI before the swap.
"""

from __future__ import annotations

import importlib.util
import io
from typing import TYPE_CHECKING, NoReturn

from fastaiagent._internal.errors import MissingPDFBackendError

if TYPE_CHECKING:
    from pypdfium2 import PdfDocument

# pypdfium2's render() takes a scale factor relative to PDF user space, which
# is 72 units per inch. dpi/72 is therefore the scale for a target DPI.
_PDF_USERSPACE_DPI = 72.0


def available() -> bool:
    """True if a local PDF engine is installed. Never imports it."""
    return importlib.util.find_spec("pypdfium2") is not None


def require(operation: str) -> NoReturn:
    """Raise an actionable :class:`MissingPDFBackendError` for ``operation``."""
    raise MissingPDFBackendError(
        f"{operation} needs a local PDF engine, which is not installed. Either:\n"
        '  * pip install "fastaiagent[pdf]"   — adds pypdfium2 (BSD-3/Apache-2.0); or\n'
        "  * bring your own parser: PDF.from_file(path, text=my_parser.extract(path))\n"
        "     — then extract_text() returns your text and no engine is needed; or\n"
        '  * use pdf_mode="native" (the default for vision-capable models), which\n'
        "     forwards the raw PDF to the provider and parses nothing locally."
    )


def _open(data: bytes) -> PdfDocument:
    import pypdfium2

    return pypdfium2.PdfDocument(data)


def page_count(data: bytes) -> int:
    """Number of pages in ``data``."""
    doc = _open(data)
    try:
        return len(doc)
    finally:
        doc.close()


def extract_text_per_page(data: bytes) -> list[str]:
    """Text of each page, one entry per page, in page order.

    Callers that need page numbers must use this rather than splitting
    :func:`extract_text` on blank lines — a single page's text can itself
    contain blank lines, which would both over-count pages and misnumber them.
    """
    doc = _open(data)
    try:
        pages: list[str] = []
        for i in range(len(doc)):
            textpage = doc[i].get_textpage()
            try:
                pages.append(textpage.get_text_range())
            finally:
                textpage.close()
        return pages
    finally:
        doc.close()


def extract_text(data: bytes) -> str:
    """Text of every page, joined with blank lines."""
    return "\n\n".join(extract_text_per_page(data))


def render_pages(data: bytes, *, dpi: int, limit: int | None = None) -> list[bytes]:
    """Render pages to PNG bytes, at most ``limit`` of them."""
    doc = _open(data)
    try:
        total = len(doc)
        count = total if limit is None else min(total, limit)
        out: list[bytes] = []
        for i in range(count):
            bitmap = doc[i].render(scale=dpi / _PDF_USERSPACE_DPI)
            buf = io.BytesIO()
            bitmap.to_pil().save(buf, format="PNG")
            out.append(buf.getvalue())
        return out
    finally:
        doc.close()
