"""MCPTool — connects to an MCP server via JSON-RPC 2.0."""

from __future__ import annotations

from typing import Any

from fastaiagent._internal.errors import ToolExecutionError
from fastaiagent.tool.base import Tool, ToolResult


class MCPTool(Tool):
    """A tool backed by an MCP (Model Context Protocol) server.

    Communicates via JSON-RPC 2.0 over HTTP.

    Example:
        tool = MCPTool(
            name="file_search",
            server_url="http://localhost:3000",
            tool_name="search_files",
        )
    """

    def __init__(
        self,
        name: str,
        server_url: str = "",
        tool_name: str = "",
        auth_token: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ):
        self.server_url = server_url
        self.tool_name = tool_name or name
        self.auth_token = auth_token
        super().__init__(name=name, description=description, parameters=parameters)

    async def _send_jsonrpc(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 request to the MCP server."""
        import httpx

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self.server_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            raise ToolExecutionError(f"MCP error: {data['error'].get('message', 'Unknown')}")
        result: dict[str, Any] = data.get("result", {})
        return result

    async def discover_tools(self) -> list[dict[str, Any]]:
        """List available tools on the MCP server."""
        result = await self._send_jsonrpc("tools/list")
        tools: list[dict[str, Any]] = result.get("tools", [])
        return tools

    async def aexecute(self, arguments: dict[str, Any], context: Any | None = None) -> ToolResult:
        """Execute the tool via MCP server."""
        try:
            result = await self._send_jsonrpc(
                "tools/call",
                {"name": self.tool_name, "arguments": arguments},
            )

            # Parse MCP content blocks
            content_parts = []
            for block in result.get("content", []):
                if block.get("type") == "text":
                    content_parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    content_parts.append(f"[image: {block.get('mimeType', 'unknown')}]")

            output = "\n".join(content_parts) if content_parts else result
            is_error = result.get("isError", False)

            if is_error:
                return ToolResult(error=str(output))
            return ToolResult(output=output)
        except ToolExecutionError:
            raise
        except Exception as e:
            raise ToolExecutionError(f"MCP tool '{self.name}' failed: {e}") from e

    def _tool_type(self) -> str:
        return "mcp"

    def _config_dict(self) -> dict[str, Any]:
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
            server_url=config.get("server_url", ""),
            tool_name=config.get("tool_name", ""),
        )
