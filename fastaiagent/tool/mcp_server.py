# mypy: disable-error-code="untyped-decorator"
"""Expose an :class:`fastaiagent.agent.Agent` or
:class:`fastaiagent.chain.Chain` as an MCP (Model Context Protocol) server.

Once a target is served, any MCP-compatible runtime — Claude Desktop,
Cursor, Continue, Zed, another fastaiagent agent via ``MCPTool`` — can use
it as a tool.

.. note::
   The upstream ``mcp`` package ships without ``py.typed`` markers, so the
   handler-registration decorators it exposes are untyped. We suppress that
   one mypy code at file scope rather than sprinkling ignores on every
   handler.

Quickstart (stdio, registers with Claude Desktop)::

    from fastaiagent import Agent, LLMClient

    agent = Agent(
        name="research_assistant",
        llm=LLMClient(provider="openai", model="gpt-4o"),
        tools=[kb.as_tool(), web_search],
    )

    if __name__ == "__main__":
        import asyncio
        asyncio.run(agent.as_mcp_server(transport="stdio").run())

and in ``~/Library/Application Support/Claude/claude_desktop_config.json``::

    {
      "mcpServers": {
        "research-assistant": {
          "command": "python",
          "args": ["/absolute/path/to/my_agent.py"]
        }
      }
    }

Installation::

    pip install 'fastaiagent[mcp-server]'

The MVP ships **stdio** transport. SSE and streamable-HTTP transports are
tracked as follow-ups.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from fastaiagent.agent.agent import Agent
    from fastaiagent.chain.chain import Chain
    from fastaiagent.tool.base import Tool

__all__ = ["FastAIAgentMCPServer", "Transport"]

Transport = Literal["stdio", "sse", "streamable-http"]


def _require_mcp() -> Any:
    """Import the upstream ``mcp`` package with an actionable error if missing."""
    try:
        import mcp.types as mcp_types
        from mcp.server import Server
    except ImportError as err:  # pragma: no cover — exercised only without the extra
        raise ImportError(
            "MCP server support requires the 'mcp' package. Install with: "
            "pip install 'fastaiagent[mcp-server]'"
        ) from err
    return Server, mcp_types


class FastAIAgentMCPServer:
    """Wrap a fastaiagent :class:`Agent` or :class:`Chain` as an MCP server.

    Args:
        target: The :class:`Agent` or :class:`Chain` to expose.
        transport: ``"stdio"`` (default; used by Claude Desktop / Cursor /
            Continue / Zed local configs). ``"sse"`` and ``"streamable-http"``
            are accepted but not yet implemented — they raise
            :class:`NotImplementedError` on :meth:`run`.
        expose_tools: If ``True`` and ``target`` is an :class:`Agent`, also
            expose each of the agent's own tools as individual MCP tools.
            Default ``False`` — keeps the surface to a single ``<target_name>``
            tool for predictability.
        expose_system_prompt: If ``True`` and ``target`` is an :class:`Agent`,
            expose its system prompt via MCP's Prompts mechanism. Default
            ``True``.
        tool_name: Override the primary tool name. Default is
            ``target.name`` (with characters not in ``[A-Za-z0-9_]`` replaced
            by ``_`` so the name is valid per the MCP spec).
        tool_description: Override the primary tool description.

    Example::

        FastAIAgentMCPServer(agent).run()                # stdio, default
        agent.as_mcp_server(expose_tools=True).run()     # also surface inner tools
    """

    def __init__(
        self,
        target: Agent | Chain,
        transport: Transport = "stdio",
        expose_tools: bool = False,
        expose_system_prompt: bool = True,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ):
        server_cls, mcp_types = _require_mcp()
        self._mcp_types = mcp_types
        self.target = target
        self.transport: Transport = transport
        self.expose_tools = expose_tools
        self.expose_system_prompt = expose_system_prompt

        target_name = getattr(target, "name", "fastaiagent")
        self._primary_name = _sanitize_name(tool_name or target_name)
        self._primary_description = (
            tool_description
            or _default_description_for(target)
        )

        # Agent has tools; Chain does not (its nodes may have tools but we
        # don't surface them individually by default).
        self._inner_tools: list[Tool] = list(getattr(target, "tools", []))

        self._server = server_cls(self._primary_name)
        self._register_handlers()

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        types = self._mcp_types

        @self._server.list_tools()
        async def handle_list_tools() -> list[Any]:
            tools = [self._primary_tool_definition()]
            if self.expose_tools:
                for t in self._inner_tools:
                    # Guard against name collisions with the primary tool.
                    if t.name == self._primary_name:
                        continue
                    tools.append(
                        types.Tool(
                            name=_sanitize_name(t.name),
                            description=t.description or t.name,
                            inputSchema=_params_to_input_schema(t.parameters),
                        )
                    )
            return tools

        @self._server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
            arguments = arguments or {}
            if name == self._primary_name:
                output_text = await self._invoke_primary(arguments)
            elif self.expose_tools:
                output_text = await self._invoke_inner_tool(name, arguments)
            else:
                raise ValueError(f"Unknown tool: {name!r}")
            return [types.TextContent(type="text", text=output_text)]

        if self.expose_system_prompt and hasattr(self.target, "system_prompt"):
            prompt_name = f"{self._primary_name}_system"

            @self._server.list_prompts()
            async def handle_list_prompts() -> list[Any]:
                return [
                    types.Prompt(
                        name=prompt_name,
                        description=f"System prompt used by the {self._primary_name!r} agent.",
                        arguments=[],
                    )
                ]

            @self._server.get_prompt()
            async def handle_get_prompt(
                name: str, arguments: dict[str, str] | None = None
            ) -> Any:
                if name != prompt_name:
                    raise ValueError(f"Unknown prompt: {name!r}")
                prompt_text = _resolve_system_prompt(self.target)
                return types.GetPromptResult(
                    description=f"System prompt for {self._primary_name!r}.",
                    messages=[
                        types.PromptMessage(
                            role="user",  # MCP only allows user/assistant
                            content=types.TextContent(type="text", text=prompt_text),
                        )
                    ],
                )

    # ------------------------------------------------------------------
    # Tool definitions & invocation
    # ------------------------------------------------------------------

    def _primary_tool_definition(self) -> Any:
        """MCP tool wrapping the whole target (agent or chain)."""
        types = self._mcp_types
        return types.Tool(
            name=self._primary_name,
            description=self._primary_description,
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": (
                            "The request to send to the agent/chain. The agent "
                            "runs to completion (including any tool calls it "
                            "makes) and returns its final text output."
                        ),
                    }
                },
                "required": ["input"],
            },
        )

    async def _invoke_primary(self, arguments: dict[str, Any]) -> str:
        user_input = str(arguments.get("input", "")).strip()
        if not user_input:
            return "Error: 'input' argument is required and cannot be empty."

        if _is_agent(self.target):
            # Avoid OTel tracing inside the MCP server loop — callers that
            # want traces can configure them on their own side.
            result = await self.target.arun(user_input, trace=False)  # type: ignore[union-attr]
            return result.output or ""
        if _is_chain(self.target):
            result = await self.target.aexecute(  # type: ignore[union-attr]
                {"input": user_input}, trace=False
            )
            # Chain output may be a str or a dict; stringify for MCP.
            output = result.output
            if isinstance(output, str):
                return output
            return json.dumps(output, default=str)
        raise TypeError(f"Unsupported target type: {type(self.target).__name__}")

    async def _invoke_inner_tool(self, name: str, arguments: dict[str, Any]) -> str:
        for t in self._inner_tools:
            if _sanitize_name(t.name) == name:
                tool_result = await t.aexecute(arguments)
                if tool_result.error:
                    return f"Error: {tool_result.error}"
                out = tool_result.output
                if isinstance(out, str):
                    return out
                return json.dumps(out, default=str)
        raise ValueError(f"Unknown tool: {name!r}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the server on the configured transport and block.

        For ``stdio`` this blocks until stdin closes (which is what Claude
        Desktop and other local MCP hosts do on shutdown).
        """
        if self.transport == "stdio":
            await self._run_stdio()
        elif self.transport in ("sse", "streamable-http"):
            raise NotImplementedError(
                f"MCP transport {self.transport!r} is not yet implemented in "
                "fastaiagent. Only 'stdio' is shipped in this release; SSE "
                "and streamable-HTTP are tracked as follow-ups."
            )
        else:
            raise ValueError(f"Unknown MCP transport: {self.transport!r}")

    async def _run_stdio(self) -> None:
        from mcp.server.stdio import stdio_server

        init_options = self._server.create_initialization_options()
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(read_stream, write_stream, init_options)

    # ------------------------------------------------------------------
    # Introspection (useful for tests)
    # ------------------------------------------------------------------

    @property
    def primary_name(self) -> str:
        return self._primary_name

    def describe(self) -> dict[str, Any]:
        """Return a dict describing what the server will expose. Handy for
        tests and for building server-capabilities dashboards without
        actually starting the stdio loop.
        """
        tools = [
            {
                "name": self._primary_name,
                "description": self._primary_description,
                "primary": True,
            }
        ]
        if self.expose_tools:
            for t in self._inner_tools:
                sanitized = _sanitize_name(t.name)
                if sanitized == self._primary_name:
                    continue
                tools.append(
                    {
                        "name": sanitized,
                        "description": t.description or t.name,
                        "primary": False,
                    }
                )
        return {
            "target_name": getattr(self.target, "name", "?"),
            "transport": self.transport,
            "expose_system_prompt": self.expose_system_prompt,
            "tools": tools,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_name(name: str) -> str:
    """Make a tool/prompt name safe for MCP (ASCII, underscores, no hyphens in some clients)."""
    out_chars: list[str] = []
    for ch in name:
        out_chars.append(ch if ch.isalnum() or ch == "_" else "_")
    sanitized = "".join(out_chars) or "fastaiagent"
    # MCP spec allows [a-zA-Z0-9_-] in practice; we already collapsed hyphens
    # to underscores for maximum client compatibility.
    return sanitized


def _is_agent(obj: Any) -> bool:
    # Lazy import to avoid circularity at module load time.
    from fastaiagent.agent.agent import Agent

    return isinstance(obj, Agent)


def _is_chain(obj: Any) -> bool:
    from fastaiagent.chain.chain import Chain

    return isinstance(obj, Chain)


def _resolve_system_prompt(target: Any) -> str:
    """Return the target's system prompt as a string, or an empty string."""
    sp = getattr(target, "system_prompt", "") or ""
    if callable(sp):
        try:
            return str(sp(None))
        except Exception:
            return ""
    return str(sp)


def _default_description_for(target: Any) -> str:
    base = getattr(target, "name", "fastaiagent target")
    sp = _resolve_system_prompt(target)
    if sp:
        first_line = sp.strip().splitlines()[0][:180]
        return f"Invoke the {base!r} agent/chain. {first_line}"
    return f"Invoke the {base!r} agent/chain. Pass your request as 'input'."


def _params_to_input_schema(parameters: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize fastaiagent tool parameters to an MCP-friendly inputSchema."""
    if not parameters:
        return {"type": "object", "properties": {}}
    # Our tools already store JSON-Schema-shaped parameter dicts, so this is
    # a passthrough — we just ensure top-level ``type: object``.
    if "type" not in parameters:
        return {"type": "object", **parameters}
    return parameters


# ---------------------------------------------------------------------------
# Factory helpers used by Agent.as_mcp_server / Chain.as_mcp_server
# ---------------------------------------------------------------------------


def _factory_for(target: Agent | Chain) -> Callable[..., FastAIAgentMCPServer]:
    """Return a thin bound factory so ``agent.as_mcp_server(...)`` reads naturally."""

    def _factory(
        transport: Transport = "stdio",
        expose_tools: bool = False,
        expose_system_prompt: bool = True,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> FastAIAgentMCPServer:
        return FastAIAgentMCPServer(
            target=target,
            transport=transport,
            expose_tools=expose_tools,
            expose_system_prompt=expose_system_prompt,
            tool_name=tool_name,
            tool_description=tool_description,
        )

    return _factory
