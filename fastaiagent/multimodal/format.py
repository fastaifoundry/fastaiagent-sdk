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
from typing import Any

from fastaiagent._internal.errors import MultimodalError, NonVisionModelError
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
            ``"native"`` (emit a single ``document`` block, Anthropic only).
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
    # image bytes to the model. PDFs in text mode don't need vision.
    needs_vision = any(
        isinstance(p, Image) or (isinstance(p, PDF) and resolved_pdf_mode != "text") for p in parts
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
    return blocks


def _pdf_blocks_openai(
    pdf: PDF, pdf_mode: str, max_pdf_pages: int, image_cap_mb: float
) -> list[dict[str, Any]]:
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
    return blocks


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
    return blocks


def _pdf_blocks_bedrock(
    pdf: PDF, pdf_mode: str, max_pdf_pages: int, image_cap_mb: float
) -> list[dict[str, Any]]:
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
