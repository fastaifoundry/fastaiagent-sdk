"""Document ingestion for LocalKB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document ready for chunking and embedding."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = ""


def ingest_file(path: str | Path) -> list[Document]:
    """Ingest a file into documents. Supports txt, md, pdf (optional)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        content = path.read_text(encoding="utf-8")
        return [Document(content=content, source=str(path), metadata={"type": suffix[1:]})]
    elif suffix == ".pdf":
        return _ingest_pdf(path)
    else:
        # Fallback: try reading as text
        content = path.read_text(encoding="utf-8", errors="replace")
        return [Document(content=content, source=str(path))]


def _ingest_pdf(path: Path) -> list[Document]:
    """Ingest a PDF file using PyMuPDF (optional dependency)."""
    try:
        import pymupdf  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "PDF ingestion requires pymupdf. Install with: pip install fastaiagent[kb]"
        )

    docs = []
    doc = pymupdf.open(str(path))
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            docs.append(
                Document(
                    content=text,
                    source=str(path),
                    metadata={"page": page_num + 1, "type": "pdf"},
                )
            )
    doc.close()
    return docs
