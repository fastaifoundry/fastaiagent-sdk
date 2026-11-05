"""How a knowledge base presents itself as a tool — shared by ``LocalKB`` and
``PlatformKB`` so the two stay interchangeable (1.99.0)."""

from __future__ import annotations

import re

#: Model providers accept tool names matching ``^[a-zA-Z0-9_-]{1,64}$``.
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def kb_tool_name(name: str) -> str:
    """``search_<name>``, made a valid tool name: other characters become ``_``
    and it is cut to 64. ``"Product docs"`` → ``search_Product_docs``. A name
    that was already valid is unchanged."""
    return f"search_{_UNSAFE.sub('_', name) or 'kb'}"[:64]


def kb_tool_description(name: str, description: str | None, *, platform: bool) -> str:
    """What the model reads to choose between knowledge bases. Without a
    ``description`` it names the KB only — which, for a ``PlatformKB`` with no
    ``name``, is its id."""
    where = "platform knowledge base" if platform else "knowledge base"
    head = f"Search the '{name}' {where}"
    return f"{head}: {description}" if description else head
