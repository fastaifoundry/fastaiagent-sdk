"""Recursive text chunking for knowledge base."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A text chunk with metadata."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    index: int = 0
    start_char: int = 0
    end_char: int = 0


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Recursively split text into chunks."""
    if not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", " "]
    chunks = _recursive_split(text, separators, chunk_size)

    result = []
    pos = 0
    for i, chunk_text_item in enumerate(chunks):
        start = text.find(chunk_text_item, max(0, pos - overlap))
        if start == -1:
            start = pos
        end = start + len(chunk_text_item)
        result.append(
            Chunk(
                content=chunk_text_item,
                metadata=metadata or {},
                index=i,
                start_char=start,
                end_char=end,
            )
        )
        pos = end

    return result


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Split text recursively using separators in order."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if len(separators) > 1 else []

    if sep and sep in text:
        parts = text.split(sep)
    else:
        if remaining_seps:
            return _recursive_split(text, remaining_seps, chunk_size)
        # Hard split
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    # Merge small parts
    chunks = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size and remaining_seps:
                chunks.extend(_recursive_split(part, remaining_seps, chunk_size))
            elif len(part) > chunk_size:
                for i in range(0, len(part), chunk_size):
                    chunk = part[i : i + chunk_size].strip()
                    if chunk:
                        chunks.append(chunk)
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks
