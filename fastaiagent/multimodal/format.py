"""``format_multimodal_message`` — convert ContentParts into a provider-shaped message.

This is the only place in the SDK that knows about provider-specific wire
formats for multimodal user messages. Every ``LLMClient._call_<provider>``
funnels list-content user messages through here.

The function is **pure**: same inputs → same outputs, no I/O beyond what
PDF rendering needs (and that's deterministic too). This is what makes
Test #4 a fast, real-library unit test rather than an HTTP-mocked one.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any
from urllib.parse import urlparse

from fastaiagent._internal.errors import MultimodalError, NonVisionModelError
from fastaiagent.multimodal.file import File
from fastaiagent.multimodal.image import Image
from fastaiagent.multimodal.pdf import PDF
from fastaiagent.multimodal.registry import supports_native_pdf
from fastaiagent.multimodal.resize import DEFAULT_MAX_IMAGE_MB, maybe_resize
from fastaiagent.multimodal.types import ContentPart

logger = logging.getLogger(__name__)

DEFAULT_MAX_PDF_PAGES: int = 20

# Per-provider per-image size ceilings (MB). Below the public limits to
# leave headroom for base64 inflation (~33%).
_PROVIDER_IMAGE_LIMIT_MB: dict[str, float] = {
    "openai": 18.0,
    "azure": 18.0,
    "anthropic": 4.5,
    "bedrock": 4.5,
    "ollama": 18.0,
    "custom": DEFAULT_MAX_IMAGE_MB,
}

_BEDROCK_IMAGE_FORMAT_FROM_MEDIA_TYPE: dict[str, str] = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}

# OpenAI ``input_audio`` accepts only these two container formats.
_OPENAI_AUDIO_FORMAT_FROM_MIME: dict[str, str] = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
}

# Bedrock Converse ``document`` block formats, keyed by mime type.
_BEDROCK_DOC_FORMAT_FROM_MIME: dict[str, str] = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/plain": "txt",
    "text/markdown": "md",
}


def _file_as_image(f: File) -> Image:
    """View an image-category ``File`` as an :class:`Image` (reuses resize +
    per-provider image emission). Raises if the mime isn't a supported image."""
    return Image(data=f.data, media_type=f.mime_type)


def _unsupported_file(provider: str, f: File) -> MultimodalError:
    return MultimodalError(
        f"{provider} has no native input for {f.mime_type!r} "
        f"({f.category}). Extract text and pass it as a string, or use a "
        f"provider that supports this type (e.g. Gemini/Bedrock for documents)."
    )


def resolve_wire_markers(value: Any) -> Any:
    """Inverse of the wire-format encoding produced by the formatter.

    Walks a list of ``{"type": ..., ...}`` dict markers and rebuilds real
    ``Image`` / ``PDF`` instances. Used by the Local UI Replay-fork modify
    endpoint when the frontend posts a JSON list of typed parts. Pure —
    no FastAPI / no I/O — so test code can import it without pulling the
    ``[ui]`` extra.

    Strings, dicts in legacy form, and other non-list values pass through
    unchanged so existing callers keep working.

    Supported part shapes::

        {"type": "text", "text": "..."}
        {"type": "image", "data_base64": "...", "media_type": "image/jpeg",
         "source_url": ?, "detail": ?}
        {"type": "pdf",   "data_base64": "..."}
    """
    if not isinstance(value, list):
        return value

    resolved: list[Any] = []
    for part in value:
        if isinstance(part, str):
            resolved.append(part)
            continue
        if not isinstance(part, dict):
            resolved.append(part)
            continue
        kind = part.get("type")
        if kind == "text":
            resolved.append(part.get("text", ""))
        elif kind == "image" and "data_base64" in part:
            resolved.append(
                Image.from_dict(
                    {
                        "data_base64": part["data_base64"],
                        "media_type": part.get("media_type", "image/png"),
                        "source_url": part.get("source_url"),
                        "detail": part.get("detail", "auto"),
                    }
                )
            )
        elif kind == "pdf" and "data_base64" in part:
            resolved.append(PDF.from_dict({"data_base64": part["data_base64"]}))
        elif kind == "file" and ("data_base64" in part or "file_id" in part):
            resolved.append(File.from_dict(part))
        else:
            resolved.append(part)
    return resolved


def format_multimodal_message(
    parts: list[ContentPart],
    provider: str,
    *,
    model: str = "",
    pdf_mode: str = "auto",
    is_vision_capable: bool = True,
    max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
    max_image_size_mb: float | None = None,
) -> dict[str, Any]:
    """Build a provider-specific message-content dict from ``parts``.

    The returned dict is intended to be **merged** into the message dict
    (``role``/``name``/``tool_call_id`` are added by the caller). For most
    providers the dict has a single ``content`` key holding a list of blocks;
    for Ollama it has ``content`` (string) plus a top-level ``images`` list.

    Arguments:
        parts: normalized list of text/Image/PDF parts.
        provider: ``"openai"``, ``"azure"``, ``"anthropic"``, ``"ollama"``,
            ``"bedrock"``, or ``"custom"``.
        model: provider-specific model id (e.g. ``"claude-sonnet-4-6"``).
            Used to decide whether to emit Anthropic-native PDF blocks.
        pdf_mode: ``"auto"`` (resolve from capability), ``"text"`` (extract
            and inline), ``"vision"`` (render pages as images), or
            ``"native"`` (forward the raw PDF to the provider — an Anthropic
            ``document`` block or an OpenAI/Azure ``file`` content part).
        is_vision_capable: caller-supplied capability flag. When ``False``
            and any non-text part is present, raises
            :class:`NonVisionModelError`.
        max_pdf_pages: cap for vision-mode rendering. Excess pages drop
            with a warning log.
        max_image_size_mb: per-image ceiling. ``None`` selects the
            provider default.
    """
    resolved_pdf_mode = _resolve_pdf_mode(pdf_mode, is_vision_capable, provider, model)
    # Vision is required only when the resolved plan would actually emit
    # image bytes to the model. Neither text mode (extracts text) nor native
    # mode (forwards raw PDF bytes for the provider to parse) needs vision.
    needs_vision = any(
        isinstance(p, Image)
        or (isinstance(p, PDF) and resolved_pdf_mode not in ("text", "native"))
        or (isinstance(p, File) and p.category == "image")
        for p in parts
    )
    if needs_vision and not is_vision_capable:
        raise NonVisionModelError(provider=provider, model=model)

    image_cap = (
        max_image_size_mb
        if max_image_size_mb is not None
        else _PROVIDER_IMAGE_LIMIT_MB.get(provider, DEFAULT_MAX_IMAGE_MB)
    )

    if provider in ("openai", "azure", "custom"):
        blocks = _openai_blocks(parts, resolved_pdf_mode, max_pdf_pages, image_cap)
        # Collapse text-only block arrays back to a plain string so non-vision
        # OpenAI models (e.g. gpt-3.5-turbo) accept the request — they reject
        # the array-content shape outright.
        if all(b.get("type") == "text" for b in blocks):
            return {"content": "\n\n".join(b.get("text", "") for b in blocks)}
        return {"content": blocks}
    if provider == "anthropic":
        return {"content": _anthropic_blocks(parts, resolved_pdf_mode, max_pdf_pages, image_cap)}
    if provider == "ollama":
        return _ollama_dict(parts, resolved_pdf_mode, max_pdf_pages, image_cap)
    if provider == "bedrock":
        return {"content": _bedrock_blocks(parts, resolved_pdf_mode, max_pdf_pages, image_cap)}
    raise MultimodalError(f"unsupported provider for multimodal formatting: {provider!r}")


# --- mode resolution ---


def _resolve_pdf_mode(mode: str, is_vision_capable: bool, provider: str, model: str) -> str:
    if mode == "auto":
        if supports_native_pdf(provider, model):
            return "native"
        return "vision" if is_vision_capable else "text"
    if mode == "native" and not supports_native_pdf(provider, model):
        # Escape hatch: ``azure`` deployments carry user-chosen names and
        # ``custom`` points at arbitrary OpenAI-compatible endpoints, so the
        # prefix registry can't vouch for them. Trust an explicit request there
        # rather than silently downgrading to vision — the integrator knows
        # their endpoint. All emit the OpenAI ``file`` part.
        if provider in ("azure", "custom"):
            return "native"
        logger.warning(
            "pdf_mode='native' requested but %s/%s does not support it; falling back to vision",
            provider,
            model,
        )
        return "vision" if is_vision_capable else "text"
    return mode


# --- provider builders ---


def _openai_blocks(
    parts: list[ContentPart],
    pdf_mode: str,
    max_pdf_pages: int,
    image_cap_mb: float,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
        elif isinstance(part, Image):
            img = maybe_resize(part, max_mb=image_cap_mb)
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.media_type};base64,{img.to_base64()}",
                        "detail": img.detail,
                    },
                }
            )
        elif isinstance(part, PDF):
            blocks.extend(_pdf_blocks_openai(part, pdf_mode, max_pdf_pages, image_cap_mb))
        elif isinstance(part, File):
            blocks.extend(_file_blocks_openai(part, image_cap_mb))
    return blocks


def _file_blocks_openai(f: File, image_cap_mb: float) -> list[dict[str, Any]]:
    if f.category == "image":
        img = maybe_resize(_file_as_image(f), max_mb=image_cap_mb)
        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{img.media_type};base64,{img.to_base64()}"},
            }
        ]
    if f.category == "audio":
        fmt = _OPENAI_AUDIO_FORMAT_FROM_MIME.get(f.mime_type)
        if fmt is None:
            raise MultimodalError(
                f"OpenAI input_audio accepts only wav/mp3; got {f.mime_type!r}"
            )
        return [{"type": "input_audio", "input_audio": {"data": f.to_base64(), "format": fmt}}]
    # An uploaded file works for any type.
    if f.file_id:
        return [{"type": "file", "file": {"file_id": f.file_id}}]
    # Inline `file_data` on Chat Completions is PDF-only; other documents must
    # be uploaded first (Files API) and referenced by file_id.
    if f.category == "pdf":
        return [
            {
                "type": "file",
                "file": {
                    "filename": f.filename or "document.pdf",
                    "file_data": f"data:{f.mime_type};base64,{f.to_base64()}",
                },
            }
        ]
    if f.category == "document":
        raise MultimodalError(
            f"OpenAI Chat Completions only takes PDF inline; upload {f.mime_type!r} "
            "via the Files API and pass File.from_file_id(<id>), or extract its text."
        )
    raise _unsupported_file("OpenAI", f)


def _pdf_filename(pdf: PDF) -> str:
    """Best-effort filename for the OpenAI ``file`` part.

    OpenAI requires a ``filename``; derive it from the PDF's origin, falling
    back to a generic name for ``PDF.from_bytes`` inputs that carry neither a
    path nor a URL.
    """
    if pdf.source_path:
        name = os.path.basename(pdf.source_path)
        if name:
            return name
    if pdf.source_url:
        name = os.path.basename(urlparse(pdf.source_url).path)
        if name:
            return name
    return "document.pdf"


def _pdf_blocks_openai(
    pdf: PDF, pdf_mode: str, max_pdf_pages: int, image_cap_mb: float
) -> list[dict[str, Any]]:
    if pdf_mode == "native":
        # Forward the raw PDF; OpenAI/Azure parse it server-side (text + page
        # images) — no local PyMuPDF rendering, so PDFs that PyMuPDF can't
        # decompress still work, matching the raw OpenAI SDK.
        return [
            {
                "type": "file",
                "file": {
                    "filename": _pdf_filename(pdf),
                    "file_data": f"data:application/pdf;base64,{pdf.to_base64()}",
                },
            }
        ]
    if pdf_mode == "text":
        return [{"type": "text", "text": pdf.extract_text()}]
    pages = pdf.to_page_images(max_pages=max_pdf_pages)
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{maybe_resize(p, max_mb=image_cap_mb).media_type};base64,"
                f"{maybe_resize(p, max_mb=image_cap_mb).to_base64()}",
                "detail": p.detail,
            },
        }
        for p in pages
    ]


def _anthropic_blocks(
    parts: list[ContentPart],
    pdf_mode: str,
    max_pdf_pages: int,
    image_cap_mb: float,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
        elif isinstance(part, Image):
            img = maybe_resize(part, max_mb=image_cap_mb)
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.media_type,
                        "data": img.to_base64(),
                    },
                }
            )
        elif isinstance(part, PDF):
            blocks.extend(_pdf_blocks_anthropic(part, pdf_mode, max_pdf_pages, image_cap_mb))
        elif isinstance(part, File):
            blocks.extend(_file_blocks_anthropic(part, image_cap_mb))
    return blocks


def _file_blocks_anthropic(f: File, image_cap_mb: float) -> list[dict[str, Any]]:
    if f.category == "image":
        img = maybe_resize(_file_as_image(f), max_mb=image_cap_mb)
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": img.to_base64(),
                },
            }
        ]
    if f.file_id:
        return [{"type": "document", "source": {"type": "file", "file_id": f.file_id}}]
    if f.category == "pdf":
        return [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": f.to_base64(),
                },
            }
        ]
    # Anthropic takes plain-text documents inline as a text source.
    if f.mime_type in ("text/plain", "text/markdown", "text/csv", "text/html"):
        return [
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": f.data.decode("utf-8", errors="replace"),
                },
            }
        ]
    raise _unsupported_file("Anthropic", f)


def _pdf_blocks_anthropic(
    pdf: PDF, pdf_mode: str, max_pdf_pages: int, image_cap_mb: float
) -> list[dict[str, Any]]:
    if pdf_mode == "native":
        return [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf.to_base64(),
                },
            }
        ]
    if pdf_mode == "text":
        return [{"type": "text", "text": pdf.extract_text()}]
    pages = pdf.to_page_images(max_pages=max_pdf_pages)
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": maybe_resize(p, max_mb=image_cap_mb).media_type,
                "data": maybe_resize(p, max_mb=image_cap_mb).to_base64(),
            },
        }
        for p in pages
    ]


def _ollama_dict(
    parts: list[ContentPart],
    pdf_mode: str,
    max_pdf_pages: int,
    image_cap_mb: float,
) -> dict[str, Any]:
    text_chunks: list[str] = []
    image_b64: list[str] = []
    for part in parts:
        if isinstance(part, str):
            text_chunks.append(part)
        elif isinstance(part, Image):
            img = maybe_resize(part, max_mb=image_cap_mb)
            image_b64.append(img.to_base64())
        elif isinstance(part, PDF):
            if pdf_mode == "text":
                text_chunks.append(part.extract_text())
            else:
                for page in part.to_page_images(max_pages=max_pdf_pages):
                    image_b64.append(maybe_resize(page, max_mb=image_cap_mb).to_base64())
        elif isinstance(part, File):
            if part.category != "image":
                raise _unsupported_file("Ollama", part)
            image_b64.append(maybe_resize(_file_as_image(part), max_mb=image_cap_mb).to_base64())
    return {"content": "\n\n".join(text_chunks), "images": image_b64}


def _bedrock_blocks(
    parts: list[ContentPart],
    pdf_mode: str,
    max_pdf_pages: int,
    image_cap_mb: float,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            blocks.append({"text": part})
        elif isinstance(part, Image):
            img = maybe_resize(part, max_mb=image_cap_mb)
            fmt = _BEDROCK_IMAGE_FORMAT_FROM_MEDIA_TYPE.get(img.media_type)
            if fmt is None:
                raise MultimodalError(
                    f"Bedrock does not accept image media_type {img.media_type!r}"
                )
            blocks.append({"image": {"format": fmt, "source": {"bytes": img.data}}})
        elif isinstance(part, PDF):
            blocks.extend(_pdf_blocks_bedrock(part, pdf_mode, max_pdf_pages, image_cap_mb))
        elif isinstance(part, File):
            blocks.extend(_file_blocks_bedrock(part, image_cap_mb))
    return blocks


def _file_blocks_bedrock(f: File, image_cap_mb: float) -> list[dict[str, Any]]:
    if f.category == "image":
        img = maybe_resize(_file_as_image(f), max_mb=image_cap_mb)
        fmt = _BEDROCK_IMAGE_FORMAT_FROM_MEDIA_TYPE.get(img.media_type)
        if fmt is None:
            raise MultimodalError(f"Bedrock does not accept image media_type {img.media_type!r}")
        return [{"image": {"format": fmt, "source": {"bytes": img.data}}}]
    doc_fmt = _BEDROCK_DOC_FORMAT_FROM_MIME.get(f.mime_type)
    if doc_fmt is not None:
        name = _bedrock_name_from(f.filename)
        return [{"document": {"format": doc_fmt, "name": name, "source": {"bytes": f.data}}}]
    raise _unsupported_file("Bedrock", f)


def _bedrock_name_from(filename: str | None) -> str:
    base = os.path.splitext(os.path.basename(filename or ""))[0] or "document"
    name = re.sub(r"[^A-Za-z0-9\s\-()\[\]]", " ", base)
    return re.sub(r"\s+", " ", name).strip() or "document"


def _bedrock_doc_name(pdf: PDF) -> str:
    """Bedrock Converse requires a document ``name`` (alphanumerics, spaces,
    hyphens, parens, brackets; no consecutive spaces). Derive from the source
    filename, else a generic name."""
    return _bedrock_name_from(pdf.source_path)


def _pdf_blocks_bedrock(
    pdf: PDF, pdf_mode: str, max_pdf_pages: int, image_cap_mb: float
) -> list[dict[str, Any]]:
    if pdf_mode == "native":
        # Bedrock Converse accepts a native document block (raw bytes) — the
        # model (Claude on Bedrock) parses the PDF server-side, no local render.
        return [
            {
                "document": {
                    "format": "pdf",
                    "name": _bedrock_doc_name(pdf),
                    "source": {"bytes": pdf.data},
                }
            }
        ]
    if pdf_mode == "text":
        return [{"text": pdf.extract_text()}]
    pages = pdf.to_page_images(max_pages=max_pdf_pages)
    return [
        {
            "image": {
                "format": "png",
                "source": {"bytes": maybe_resize(p, max_mb=image_cap_mb).data},
            }
        }
        for p in pages
    ]
