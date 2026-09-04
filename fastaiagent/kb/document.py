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
    """Ingest a PDF file, one Document per non-empty page.

    Needs a local PDF engine (optional dependency). ``fastaiagent[kb]`` pulls
    it in; ``fastaiagent[pdf]`` is the lighter option if you only want PDF
    decoding without the embedding stack.
    """
    from fastaiagent.multimodal import _pdf_backend

    if not _pdf_backend.available():
        raise ImportError(
            "PDF ingestion requires a local PDF engine. Install with: "
            'pip install "fastaiagent[kb]" (or the lighter "fastaiagent[pdf]"). '
            "Alternatively, extract the text yourself and pass it to "
            "LocalKB.add() as a string."
        )

    data = path.read_bytes()
    docs = []
    # Per page, so ``metadata["page"]`` is right. Splitting the joined
    # ``extract_text`` output on blank lines would misnumber any page whose
    # own text contains one.
    for page_num, text in enumerate(_pdf_backend.extract_text_per_page(data)):
        if text.strip():
            docs.append(
                Document(
                    content=text,
                    source=str(path),
                    metadata={"page": page_num + 1, "type": "pdf"},
                )
            )
    return docs
