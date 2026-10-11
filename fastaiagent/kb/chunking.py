"""Recursive text chunking for knowledge base."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A text chunk with metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
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
    """Recursively split text into chunks of at most ``chunk_size`` characters.

    The splitter cuts at the coarsest seam that fits — paragraph, line, sentence,
    word — and every chunk after the first then **starts with up to ``overlap``
    characters of the text before it**, cut at a word boundary, so a fact that
    straddles a seam appears whole in at least one chunk. To keep the
    ``chunk_size`` bound, the seams are found at ``chunk_size - overlap``.
    ``overlap`` is capped at half of ``chunk_size``.

    Until 1.88.0 ``overlap`` was accepted and had no effect on the chunk text:
    neighbouring chunks shared nothing, while the docs described a 50-character
    overlap. An index built before 1.88.0 has no overlap until it is re-ingested.
    """
    if not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", " "]
    ov = max(0, min(overlap, chunk_size // 2))
    if len(text) <= chunk_size or ov == 0:
        pieces = _recursive_split(text, separators, chunk_size)
        ov = 0
    else:
        pieces = _recursive_split(text, separators, chunk_size - ov)

    result: list[Chunk] = []
    pos = 0
    prev_start: int | None = None
    prev_piece = ""
    for i, piece in enumerate(pieces):
        found = text.find(piece, pos)
        content, start = piece, (found if found != -1 else pos)
        end = start + len(piece)
        if i > 0 and ov:
            if found != -1 and prev_start is not None:
                # The overlap is a real slice of the source, not a copy pasted in
                # front: ``start_char``/``end_char`` stay true offsets.
                ov_start = _word_start(text, max(prev_start, found - ov), found)
                if ov_start < found:
                    content, start = text[ov_start:end], ov_start
            else:
                # The piece could not be located verbatim (the splitter normalised
                # whitespace inside it). Carry the previous piece's tail instead.
                tail = prev_piece[-(ov - 1) :] if ov > 1 else ""
                cut = tail.find(" ")
                tail = tail[cut + 1 :] if cut != -1 else tail
                if tail:
                    content = f"{tail} {piece}"
        result.append(
            Chunk(
                content=content,
                metadata=metadata or {},
                index=i,
                start_char=start,
                end_char=end,
            )
        )
        prev_start = found if found != -1 else None
        prev_piece = piece
        pos = end

    if ov:
        # A short piece followed by its overlapping neighbour ends up wholly
        # inside that neighbour ("Refund policy" then "Refund policy. Items …").
        # Keeping it would index the same text twice and let it outrank the
        # chunk that holds its context, so it is dropped and the rest reindexed.
        kept = [
            c
            for c, nxt in zip(result, [*result[1:], None])
            if nxt is None or not (nxt.start_char <= c.start_char and c.end_char <= nxt.end_char)
        ]
        for i, c in enumerate(kept):
            c.index = i
        result = kept

    return result


def _word_start(text: str, at: int, limit: int) -> int:
    """The first word start at or after ``at`` and before ``limit``, else ``limit``.

    An overlap that began mid-word would hand the embedder a fragment like
    ``"ithin 30 days"``; starting at the next word keeps it readable.
    """
    if at > 0 and not text[at - 1].isspace() and not text[at].isspace():
        nxt = at
        while nxt < limit and not text[nxt].isspace():
            nxt += 1
        at = nxt
    while at < limit and text[at].isspace():
        at += 1
    return at


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
            # ``current`` has been emitted; it must not carry into what follows.
            # Before 1.88.0 the two long-part branches below left it set, so the
            # text before a part longer than ``chunk_size`` was emitted a second
            # time, glued to the next short part: a short paragraph ahead of a
            # long one came back later stuck to an unrelated paragraph.
            current = ""
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
