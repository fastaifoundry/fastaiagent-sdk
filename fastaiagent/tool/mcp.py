"""MCPTool — a tool on an MCP (Model Context Protocol) server, called through the
official ``mcp`` client library (``pip install "fastaiagent[mcp]"``).

Two ways to reach a server:

* ``server_url=`` — a remote server over streamable HTTP;
* ``command=`` — a local server the SDK starts as a subprocess (stdio).

Each call opens a session (``initialize``), calls the tool and closes it. Until
1.92.0 this module spoke a hand-written JSON-RPC dialect that official servers
reject (406 / 400); it is gone.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from fastaiagent._internal.errors import ToolExecutionError
from fastaiagent.tool.base import Tool, ToolResult

_INSTALL = 'MCPTool needs the official MCP client: pip install "fastaiagent[mcp]"'

#: How long the handshake and ``tools/list`` may take. The official client waits
#: forever for a reply by default, so a URL that is not an MCP server would hang.
_HANDSHAKE_SECONDS = 30.0

#: How long a tool call may take when the tool sets no ``timeout`` (the same as
#: the HTTP read timeout).
_CALL_SECONDS = 300.0


def _require_mcp() -> None:
    try:
        import mcp  # noqa: F401
    except ImportError as err:
        raise ImportError(_INSTALL) from err


class _GuardedTransport(httpx.AsyncBaseTransport):
    """Every request — the first and each redirect hop — passes the SDK's SSRF
    check (security_audit_2 N15) and connects to the address it checked, so a
    DNS answer that changes in between cannot redirect it (1.92.0).
    ``server_url`` can arrive from a payload (the runner, a trace,
    ``from_dict``), so cloud metadata and private networks are refused; loopback
    stays allowed for local servers. ``FASTAIAGENT_ALLOW_PRIVATE_NETWORKS=1``
    opens intranet hosts."""

    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        from fastaiagent.multimodal._http import asend_pinned, validate_url

        addresses = validate_url(str(request.url), allow_loopback=True)
        return await asend_pinned(self._inner.handle_async_request, request, addresses)

    async def aclose(self) -> None:
        await self._inner.aclose()


@asynccontextmanager
async def _session(
    server_url: str,
    auth_token: str | None,
    command: list[str] | None,
    env: dict[str, str] | None,
) -> AsyncIterator[Any]:
    """An initialized ``mcp.ClientSession`` for one server."""
    _require_mcp()
    from datetime import timedelta

    from mcp import ClientSession

    wait = timedelta(seconds=_HANDSHAKE_SECONDS)

    if command:
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(command=command[0], args=command[1:], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=wait) as session:
                await session.initialize()
                yield session
        return

    if not server_url:
        # A stdio tool's command is never serialized, so one rebuilt from a
        # payload (the runner, the plane, Replay, ``from_dict``) arrives here.
        raise ToolExecutionError(
            "MCPTool has no server_url and no command. A local (command=) MCP tool "
            "is never serialized, so it cannot run after being rebuilt from a payload."
        )

    from mcp.client.streamable_http import streamable_http_client

    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    async with httpx.AsyncClient(
        transport=_GuardedTransport(),
        headers=headers,
        timeout=httpx.Timeout(30, read=300),
        follow_redirects=True,
    ) as http:
        async with streamable_http_client(server_url, http_client=http) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write, read_timeout_seconds=wait) as session:
                await session.initialize()
                yield session


def _root_cause(err: BaseException) -> BaseException:
    """The first real error inside the task groups the MCP client runs in."""
    while getattr(err, "exceptions", None):
        err = err.exceptions[0]  # type: ignore[attr-defined]
    return err


def _to_tool_result(result: Any) -> ToolResult:
    """Map a ``CallToolResult`` onto a ``ToolResult``."""
    parts: list[str] = []
    for block in result.content:
        kind = getattr(block, "type", "")
        if kind == "text":
            parts.append(block.text)
        elif kind == "image":
            parts.append(f"[image: {block.mimeType}]")
        elif kind == "resource" and getattr(block.resource, "text", None) is not None:
            parts.append(block.resource.text)
        else:
            parts.append(f"[{kind}]")
    output: Any = "\n".join(parts) if parts else (result.structuredContent or "")
    if result.isError:
        return ToolResult(error=str(output) or "the MCP server reported an error")
    return ToolResult(output=output)


class MCPTool(Tool):
    """A tool on an MCP server.

    Example::

        # A remote server
        search = MCPTool(name="search", server_url="https://mcp.example.com/mcp",
                         auth_token="...", parameters={...})

        # Every tool a server offers, with names and schemas filled in
        tools = MCPTool.from_server(
            command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )
    """

    origin = "mcp"

    def __init__(
        self,
        name: str,
        server_url: str = "",
        tool_name: str = "",
        auth_token: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        replay_class: str | None = None,
        *,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int = 0,
        retry_delay: float = 0.5,
        output_type: Any | None = None,
        deferred: bool = False,
    ):
        if server_url and command:
            raise ValueError("MCPTool takes server_url or command, not both")
        self.server_url = server_url
        self.tool_name = tool_name or name
        self.auth_token = auth_token
        self.command = list(command) if command else None
        self.env = env
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            replay_class=replay_class,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            output_type=output_type,
            deferred=deferred,
        )

    @classmethod
    async def afrom_server(
        cls,
        server_url: str = "",
        *,
        auth_token: str | None = None,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        **tool_kwargs: Any,
    ) -> list[MCPTool]:
        """One ``MCPTool`` per tool the server lists, with its name, description
        and input schema. ``tool_kwargs`` (``timeout``, ``replay_class``, …) go
        to every tool."""
        if server_url and command:
            raise ValueError("from_server takes server_url or command, not both")
        found: list[Any] = []
        async with _session(server_url, auth_token, command, env) as session:
            cursor = None
            while True:
                page = await session.list_tools(cursor) if cursor else await session.list_tools()
                found.extend(page.tools)
                cursor = page.nextCursor
                if not cursor:
                    break
        return [
            cls(
                name=t.name,
                server_url=server_url,
                tool_name=t.name,
                auth_token=auth_token,
                description=t.description or "",
                parameters=t.inputSchema,
                command=command,
                env=env,
                **tool_kwargs,
            )
            for t in found
        ]

    @classmethod
    def from_server(
        cls,
        server_url: str = "",
        *,
        auth_token: str | None = None,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        **tool_kwargs: Any,
    ) -> list[MCPTool]:
        """Sync form of :meth:`afrom_server`."""
        from fastaiagent._internal.async_utils import run_sync

        return run_sync(
            cls.afrom_server(
                server_url, auth_token=auth_token, command=command, env=env, **tool_kwargs
            )
        )

    async def aexecute(self, arguments: dict[str, Any], context: Any | None = None) -> ToolResult:
        """Call the tool on its server."""
        from datetime import timedelta

        wait = timedelta(seconds=self.timeout or _CALL_SECONDS)
        try:
            async with _session(self.server_url, self.auth_token, self.command, self.env) as s:
                result = await s.call_tool(self.tool_name, arguments, read_timeout_seconds=wait)
        except ToolExecutionError:
            raise
        except Exception as e:
            raise ToolExecutionError(f"MCP tool '{self.name}' failed: {_root_cause(e)}") from e
        return _to_tool_result(result)

    def _tool_type(self) -> str:
        return "mcp"

    def _config_dict(self) -> dict[str, Any]:
        # ``command``/``env``/``auth_token`` are never serialized: a payload must
        # not be able to start a process or carry a credential.
        return {
            "server_url": self.server_url,
            "tool_name": self.tool_name,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MCPTool:
        config = data.get("config", {})
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parameters=data.get("parameters"),
            replay_class=data.get("replay_class", "side_effecting"),
            server_url=config.get("server_url", ""),
            tool_name=config.get("tool_name", ""),
        )
