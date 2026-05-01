"""Vision-capable model registry.

A small, authoritative catalog of which provider+model combinations support
image input. Used by the LLMClient to:

* auto-resolve ``pdf_mode="auto"`` to ``vision`` vs ``text``
* raise :class:`NonVisionModelError` early when an image is sent to a
  non-vision model (clearer than letting the provider 4xx)

Maintained as **prefixes** because vendors release new vision-capable variants
constantly (gpt-4o-2024-..., claude-sonnet-4-..., gemini-1.5-...). Prefix
matching keeps this list small without going stale every month. Explicit
non-vision overrides are listed below the matchers.
"""

from __future__ import annotations

# Each entry is a (provider, model_prefix) — model.startswith(prefix) → vision-capable.
_VISION_PREFIXES: tuple[tuple[str, str], ...] = (
    # OpenAI
    ("openai", "gpt-4o"),
    ("openai", "gpt-4-turbo"),
    ("openai", "gpt-4-vision"),
    ("openai", "gpt-5"),
    ("openai", "o1"),
    ("openai", "o3"),
    ("openai", "o4"),
    # Azure (OpenAI-compatible deployments — naming varies but follows the upstream)
    ("azure", "gpt-4o"),
    ("azure", "gpt-4-turbo"),
    ("azure", "gpt-4-vision"),
    # Anthropic
    ("anthropic", "claude-3-5"),
    ("anthropic", "claude-3-7"),
    ("anthropic", "claude-3-opus"),
    ("anthropic", "claude-3-sonnet"),
    ("anthropic", "claude-3-haiku"),
    ("anthropic", "claude-sonnet-4"),
    ("anthropic", "claude-opus-4"),
    ("anthropic", "claude-haiku-4"),
    # Bedrock-hosted Claude — ids include the AWS region prefix (anthropic.claude-...)
    ("bedrock", "anthropic.claude-3"),
    ("bedrock", "anthropic.claude-sonnet-4"),
    ("bedrock", "anthropic.claude-opus-4"),
    ("bedrock", "anthropic.claude-haiku-4"),
    # Ollama vision tags — the user opts in by tagging their pull
    ("ollama", "llava"),
    ("ollama", "bakllava"),
    ("ollama", "llama3.2-vision"),
    ("ollama", "qwen2-vl"),
    ("ollama", "qwen2.5-vl"),
    ("ollama", "moondream"),
    # Custom — caller-controlled; we trust the integrator
    ("custom", ""),
)

# Explicit non-vision overrides for misleading prefixes (e.g. "gpt-4" alone).
_NON_VISION_EXACT: frozenset[tuple[str, str]] = frozenset(
    {
        ("openai", "gpt-3.5-turbo"),
        ("openai", "gpt-4"),
        ("openai", "gpt-4-32k"),
    }
)

# Anthropic models that natively accept ``application/pdf`` document blocks
# (no client-side rendering required). Maintained as prefixes too.
_NATIVE_PDF_PREFIXES: tuple[tuple[str, str], ...] = (
    ("anthropic", "claude-3-5"),
    ("anthropic", "claude-3-7"),
    ("anthropic", "claude-sonnet-4"),
    ("anthropic", "claude-opus-4"),
    ("anthropic", "claude-haiku-4"),
    ("bedrock", "anthropic.claude-3-5"),
    ("bedrock", "anthropic.claude-sonnet-4"),
    ("bedrock", "anthropic.claude-opus-4"),
    ("bedrock", "anthropic.claude-haiku-4"),
)


def is_vision_capable(provider: str, model: str) -> bool:
    """Return ``True`` if the (provider, model) pair supports image input.

    ``custom`` is always accepted because the caller is responsible for the
    endpoint they're pointing at.
    """
    if provider == "custom":
        return True
    if (provider, model) in _NON_VISION_EXACT:
        return False
    for p, prefix in _VISION_PREFIXES:
        if p == provider and prefix and model.startswith(prefix):
            return True
    return False


def supports_native_pdf(provider: str, model: str) -> bool:
    """Return ``True`` if this provider+model accepts native PDF document blocks."""
    for p, prefix in _NATIVE_PDF_PREFIXES:
        if p == provider and model.startswith(prefix):
            return True
    return False
