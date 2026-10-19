"""Tools the model's provider runs itself (1.91.0, 2D).

:func:`web_search`, :func:`code_execution` and :func:`file_search` return a
:class:`ProviderTool`. It goes in ``Agent(tools=[...])`` like any tool, but it is
never run here: it is sent to the provider in the provider's own shape —

=================  ===========================  ========================  ==================
                   OpenAI (``api="responses"``)  Anthropic                 Gemini
=================  ===========================  ========================  ==================
``web_search``     ``web_search``               ``web_search_20250305``   ``google_search``
``code_execution`` ``code_interpreter``         ``code_execution_20250825`` ``code_execution``
``file_search``    ``file_search``              —                         ``file_search``
=================  ===========================  ========================  ==================

— and the provider runs it inside the model call. Each call it reports becomes a
``tool.<name>`` span (``tool.origin="provider"``) and is judged by the agent's
``tool_result`` guardrails. A provider that runs none of these (OpenAI Chat
Completions, Ollama, Bedrock, the OpenAI-compatible presets) refuses the request
before it is sent, rather than dropping the tool.
"""

from __future__ import annotations

from typing import Any

from fastaiagent.tool.base import Tool, ToolResult

#: The kinds a :class:`ProviderTool` can be.
PROVIDER_TOOL_KINDS = ("web_search", "code_execution", "file_search")

_DESCRIPTIONS = {
    "web_search": "Searches the web. Run by the model's provider.",
    "code_execution": "Runs code in the provider's sandbox. Run by the model's provider.",
    "file_search": "Searches the given file stores. Run by the model's provider.",
}


class ProviderTool(Tool):
    """A tool the model's provider runs — see :mod:`fastaiagent.tool.provider`.

    Build one with :func:`web_search`, :func:`code_execution` or
    :func:`file_search`. ``options`` are the SDK's own names; each provider's
    request builder puts them in its own fields and refuses one it cannot honour.
    """

    origin = "provider"

    def __init__(self, kind: str, options: dict[str, Any] | None = None):
        if kind not in PROVIDER_TOOL_KINDS:
            raise ValueError(f"kind must be one of {PROVIDER_TOOL_KINDS}, got {kind!r}")
        super().__init__(name=kind, description=_DESCRIPTIONS[kind])
        self.kind = kind
        self.options: dict[str, Any] = {k: v for k, v in (options or {}).items() if v is not None}

    async def aexecute(self, arguments: dict[str, Any], context: Any | None = None) -> ToolResult:
        # Reached only outside an agent's model call — a Chain tool node, say.
        return ToolResult(
            error=(
                f"'{self.name}' is run by the model's provider inside a model call; "
                f"give it to an Agent instead of calling it directly."
            )
        )

    def to_openai_format(self) -> dict[str, Any]:
        """A marker each provider's request builder turns into its own shape."""
        return {"type": "provider", "provider_tool": {"kind": self.kind, "options": self.options}}

    def _tool_type(self) -> str:
        return "provider"

    def _config_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "options": dict(self.options)}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ProviderTool:
        config = data.get("config") or {}
        return cls(kind=config.get("kind") or data.get("name", ""), options=config.get("options"))


def web_search(
    *,
    max_uses: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    user_location: dict[str, str] | None = None,
) -> ProviderTool:
    """Let the model search the web, run by its provider.

    Args:
        max_uses: The most searches per model call (Anthropic only).
        allowed_domains: Search only these domains (OpenAI, Anthropic).
        blocked_domains: Never search these domains (OpenAI, Anthropic; Anthropic
            takes one of the two lists, not both).
        user_location: Approximate location to localise results —
            ``{"city", "region", "country", "timezone"}`` (OpenAI, Anthropic).

    Gemini's ``google_search`` takes none of these, so an option set there is
    refused rather than ignored.
    """
    return ProviderTool(
        "web_search",
        {
            "max_uses": max_uses,
            "allowed_domains": list(allowed_domains) if allowed_domains else None,
            "blocked_domains": list(blocked_domains) if blocked_domains else None,
            "user_location": dict(user_location) if user_location else None,
        },
    )


def code_execution() -> ProviderTool:
    """Let the model run code in its provider's sandbox (Python on OpenAI and
    Gemini, Bash + files on Anthropic). Each model call may get a fresh sandbox."""
    return ProviderTool("code_execution")


def file_search(stores: list[str], *, max_results: int | None = None) -> ProviderTool:
    """Let the model search file stores you created at the provider.

    Args:
        stores: OpenAI vector store ids, or Gemini file search store names.
        max_results: The most results per search (OpenAI only).

    Anthropic has no hosted file search.
    """
    if not stores:
        raise ValueError("file_search() needs at least one store to search")
    return ProviderTool("file_search", {"stores": list(stores), "max_results": max_results})
