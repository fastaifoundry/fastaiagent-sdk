"""Shared LocalKB — the same instance is plumbed into all three sub-examples
via each integration's ``kb_as_retriever()`` (LangChain) or
``kb_as_tool()`` (CrewAI, PydanticAI). One source of truth across frameworks.

**Important — KB path coupling.** Each integration's ``kb_as_retriever()``
/ ``kb_as_tool()`` re-instantiates ``LocalKB(name=kb_name)`` *with the
default path* — they don't accept a ``path=`` kwarg. So the KB created
here MUST also live at the default path (``~/.fastaiagent/kb/<name>``)
or the integration will read from an empty store. We deliberately omit
``path=`` here for that reason. Document this gotcha; don't fight it.
"""

from __future__ import annotations

from pathlib import Path

import fastaiagent as fa

_HERE = Path(__file__).resolve().parent.parent

support_kb = fa.LocalKB(
    name="support-kb",
    chunk_size=512,
    chunk_overlap=50,
)
if support_kb.status()["chunk_count"] == 0:
    support_kb.add(str(_HERE / "knowledge"))
