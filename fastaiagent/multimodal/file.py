"""``File`` — a generic native file/bytes input for multimodal LLM calls.

Where :class:`Image` and :class:`PDF` are type-specific, ``File`` carries *any*
bytes plus a mime type and lets the per-provider formatter route it to that
provider's native file mechanism (OpenAI ``file``/``image_url``/``input_audio``
parts, Anthropic ``document``/``image`` blocks, Gemini ``inlineData``, Bedrock
``document``/``image`` blocks). The mime type is sniffed from the bytes/filename
when not given, so ``Agent.run(file_bytes)`` and ``Agent.run(Path("x.docx"))``
work without the caller naming the type.

A ``File`` can also reference a provider-uploaded file by ``file_id`` (OpenAI /
Anthropic Files API) instead of carrying bytes — useful for large or reused
files.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastaiagent._internal.errors import MultimodalError
from fastaiagent.multimodal._http import safe_http_fetch

_FROM_URL_TIMEOUT_SECONDS: float = 30.0
_FROM_URL_MAX_REDIRECTS: int = 5
_FROM_URL_MAX_BYTES: int = 100 * 1024 * 1024  # 100 MiB

_DEFAULT_MIME = "application/octet-stream"

# Document mime types that providers accept as native "document" inputs.
_DOCUMENT_MIMES: frozenset[str] = frozenset(
    {
        "text/plain",
        "text/csv",
        "text/html",
        "text/markdown",
        "application/json",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
    }
)


# ``mimetypes.guess_type`` is platform/version dependent for some text types
# (e.g. ``.md`` maps to ``text/markdown`` on Linux/py3.12+ but ``None`` on
# macOS py3.10/3.11). Pin the document extensions we care about so sniffing is
# deterministic everywhere.
_EXT_MIME_FALLBACK: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
}


def sniff_mime(data: bytes | None, filename: str | None = None) -> str:
    """Best-effort mime type from magic bytes, then filename extension.

    Returns ``application/octet-stream`` when nothing matches so callers always
    get a usable string (the formatter decides whether a provider accepts it).
    """
    if data:
        magic = _sniff_magic(data)
        if magic:
            return magic
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
        ext = os.path.splitext(filename)[1].lower()
        if ext in _EXT_MIME_FALLBACK:
            return _EXT_MIME_FALLBACK[ext]
    return _DEFAULT_MIME


def _sniff_magic(data: bytes) -> str | None:
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data.startswith(b"ID3") or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if data[4:12] == b"ftypisom" or data[4:8] == b"ftyp":
        return "video/mp4"
    return None


@dataclass
class File:
    """A generic file input for multimodal LLM calls.

    Construct via :py:meth:`from_bytes`, :py:meth:`from_path`,
    :py:meth:`from_url`, or :py:meth:`from_file_id`. The ``mime_type`` drives
    how each provider receives it — see :py:attr:`category`.
    """

    data: bytes
    mime_type: str
    filename: str | None = None
    source_path: str | None = None
    source_url: str | None = None
    file_id: str | None = None

    def __post_init__(self) -> None:
        # Either inline bytes or an uploaded file_id must be present.
        if not self.file_id and (not isinstance(self.data, bytes) or len(self.data) == 0):
            raise MultimodalError("File requires non-empty bytes (or a file_id reference)")
        if not self.mime_type:
            self.mime_type = sniff_mime(self.data or None, self.filename)

    # --- constructors ---

    @classmethod
    def from_bytes(
        cls, data: bytes, *, mime_type: str | None = None, filename: str | None = None
    ) -> File:
        return cls(
            data=data,
            mime_type=mime_type or sniff_mime(data, filename),
            filename=filename,
        )

    @classmethod
    def from_path(cls, path: str | Path, *, mime_type: str | None = None) -> File:
        p = Path(path)
        data = p.read_bytes()
        return cls(
            data=data,
            mime_type=mime_type or sniff_mime(data, p.name),
            filename=p.name,
            source_path=str(p),
        )

    @classmethod
    def from_url(cls, url: str, *, mime_type: str | None = None) -> File:
        """Fetch a file from an HTTP(S) URL (SSRF-hardened, 100 MiB cap)."""
        resp = safe_http_fetch(
            url,
            timeout=_FROM_URL_TIMEOUT_SECONDS,
            max_redirects=_FROM_URL_MAX_REDIRECTS,
            max_bytes=_FROM_URL_MAX_BYTES,
        )
        header_ct = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        name = url.rsplit("/", 1)[-1] or None
        return cls(
            data=resp.content,
            mime_type=mime_type or header_ct or sniff_mime(resp.content, name),
            filename=name,
            source_url=url,
        )

    @classmethod
    def from_file_id(
        cls, file_id: str, *, mime_type: str | None = None, filename: str | None = None
    ) -> File:
        """Reference a provider-uploaded file (OpenAI/Anthropic Files API)."""
        return cls(
            data=b"",
            mime_type=mime_type or (sniff_mime(None, filename) if filename else _DEFAULT_MIME),
            filename=filename,
            file_id=file_id,
        )

    # --- classification ---

    @property
    def category(self) -> str:
        """One of ``image``, ``audio``, ``video``, ``pdf``, ``document``, ``other``."""
        m = self.mime_type.lower()
        if m == "application/pdf":
            return "pdf"
        if m.startswith("image/"):
            return "image"
        if m.startswith("audio/"):
            return "audio"
        if m.startswith("video/"):
            return "video"
        if m.startswith("text/") or m in _DOCUMENT_MIMES:
            return "document"
        return "other"

    # --- serialization ---

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def size_bytes(self) -> int:
        return len(self.data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "file",
            "data_base64": self.to_base64() if self.data else "",
            "mime_type": self.mime_type,
            "filename": self.filename,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "file_id": self.file_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> File:
        raw = d.get("data_base64") or ""
        return cls(
            data=base64.b64decode(raw) if raw else b"",
            mime_type=d.get("mime_type") or _DEFAULT_MIME,
            filename=d.get("filename"),
            source_path=d.get("source_path"),
            source_url=d.get("source_url"),
            file_id=d.get("file_id"),
        )
