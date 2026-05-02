"""Tolerant attribute lookup for span attribute dicts.

The live SDK emits unprefixed attribute names (``agent.name``,
``agent.tokens_used``, ``agent.latency_ms``, etc.). We also support two
namespaced variants:

* ``fastaiagent.*`` — the canonical prefix going forward for anything the
  SDK adds on top of the standard OTel ``gen_ai.*`` namespace.
* ``fastai.*`` — legacy; never emit it, but accept it on read so old
  traces don't suddenly go blank after upgrading.

``fastai.*`` is reserved by fast.ai (a different company); we never write
new attributes under that prefix. See
``feedback_no_fastai_namespace.md``.

Use :func:`attr` to read any custom field:

    attr(span_attrs, "agent.name")
    # tries fastaiagent.agent.name → agent.name → fastai.agent.name
"""

from __future__ import annotations

from typing import Any


def attr(attrs: dict[str, Any] | None, key: str) -> Any:
    """Read ``key`` from a span attributes dict, tolerating prefix variants.

    Search order:
    1. ``fastaiagent.<key>`` (canonical namespaced).
    2. ``<key>`` (unprefixed — the live SDK's current convention).
    3. ``fastai.<key>`` (legacy).

    Returns the first non-``None`` value found, or ``None`` if none match.
    """
    if not attrs:
        return None
    for candidate in (f"fastaiagent.{key}", key, f"fastai.{key}"):
        value = attrs.get(candidate)
        if value is not None:
            return value
    return None


def trace_cost_usd(attrs: dict[str, Any] | None) -> float | None:
    """Best-effort cost lookup across every attribute variant.

    Checks ``agent.cost_usd`` first (what the SDK will emit once cost
    tracking lands), then falls back to the historical ``cost.total_usd``
    key under either prefix.
    """
    for key in ("agent.cost_usd", "cost.total_usd"):
        value = attr(attrs, key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# Multimodal content-part detection
# ---------------------------------------------------------------------------


def _looks_like_image_part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    typ = str(part.get("type", "")).lower()
    if typ in {"image", "input_image", "image_url"}:
        return True
    media = str(part.get("media_type", "") or "").lower()
    return media.startswith("image/")


def _looks_like_pdf_part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    typ = str(part.get("type", "")).lower()
    if typ in {"input_pdf", "pdf", "document"}:
        return True
    media = str(part.get("media_type", "") or "").lower()
    return media == "application/pdf"


def _looks_like_text_part(part: Any) -> bool:
    if isinstance(part, str):
        return True
    if not isinstance(part, dict):
        return False
    return str(part.get("type", "")).lower() in {"text", "input_text", "output_text"}


def extract_content_parts(value: Any) -> list[dict[str, Any]]:
    """Walk a JSON-decoded message/content payload and return content parts.

    Handles three common shapes the SDK and providers emit:

    * a plain string (returns ``[{"type": "text", "text": <s>}]``)
    * a list of content parts (returned in original order)
    * a dict with a ``content`` key (the OpenAI/Anthropic message shape) —
      recurses into ``content``
    * a list of messages (each with ``content``) — flattens parts in
      message order

    Used by tests and any server-side processing that needs to know
    whether a span carries images/PDFs without loading the full payload.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    parts: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "content" in value:
            return extract_content_parts(value["content"])
        if "type" in value:
            parts.append(value)
            return parts
        return []
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, dict) and "content" in entry:
                parts.extend(extract_content_parts(entry["content"]))
            elif _looks_like_text_part(entry):
                if isinstance(entry, str):
                    parts.append({"type": "text", "text": entry})
                else:
                    parts.append(entry)
            elif _looks_like_image_part(entry) or _looks_like_pdf_part(entry):
                parts.append(entry)
            elif isinstance(entry, dict):
                parts.append(entry)
        return parts
    return []


def has_multimodal_part(value: Any) -> bool:
    """True if any image or PDF content part is present in ``value``."""
    return any(
        _looks_like_image_part(p) or _looks_like_pdf_part(p)
        for p in extract_content_parts(value)
    )
