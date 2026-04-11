"""End-to-end quality gate — MCPTool against a local stub MCP server.

Spins up a minimal JSON-RPC 2.0 HTTP server in a daemon thread that
implements ``tools/list`` and ``tools/call`` well enough for the
MCPTool client to exercise its discover + call path end-to-end.

The stub is intentionally tiny — it's not a real MCP server, it's just
the smallest thing that responds in the shape MCPTool expects. That's
enough to catch regressions in the JSON-RPC client, content-block
parsing, and error-path handling without depending on an external MCP
implementation.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


class _StubMCPHandler(BaseHTTPRequestHandler):
    """Handles POST / with a JSON-RPC 2.0 payload."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence the default access log so pytest output stays clean."""
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._write_json({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            return

        method = req.get("method", "")
        req_id = req.get("id", 1)
        params = req.get("params", {}) or {}

        if method == "tools/list":
            body = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo_tool",
                            "description": "Echo the input string back.",
                        }
                    ]
                },
            }
        elif method == "tools/call":
            args = params.get("arguments", {}) or {}
            text = args.get("message", "")
            if text == "__error__":
                body = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": "simulated tool error"}],
                        "isError": True,
                    },
                }
            else:
                body = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": f"echo: {text}"}
                        ],
                        "isError": False,
                    },
                }
        else:
            body = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"unknown method {method}"},
            }

        self._write_json(body)

    def _write_json(self, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture(scope="module")
def stub_mcp_server():
    """Spin up a stub MCP server on 127.0.0.1:<ephemeral port> for the module."""
    server = HTTPServer(("127.0.0.1", 0), _StubMCPHandler)
    host, port = server.server_address
    url = f"http://{host}:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()


class TestMCPToolGate:
    def test_01_discover_tools(self, stub_mcp_server: str) -> None:
        require_env()
        from fastaiagent import MCPTool

        tool = MCPTool(
            name="echo_tool",
            server_url=stub_mcp_server,
            tool_name="echo_tool",
        )
        # discover_tools is async; use run_sync via the internal util
        from fastaiagent._internal.async_utils import run_sync

        tools = run_sync(tool.discover_tools())
        assert tools, "discover_tools returned empty list"
        assert any(t.get("name") == "echo_tool" for t in tools), (
            f"echo_tool not in discovered list: {tools}"
        )

    def test_02_call_success_path(self, stub_mcp_server: str) -> None:
        require_env()
        from fastaiagent import MCPTool

        tool = MCPTool(
            name="echo_tool",
            server_url=stub_mcp_server,
            tool_name="echo_tool",
        )
        result = tool.execute({"message": "hello mcp"})
        assert result.success, f"MCP tool errored: {result.error}"
        assert isinstance(result.output, str)
        assert "echo: hello mcp" in result.output

    def test_03_call_error_path(self, stub_mcp_server: str) -> None:
        """Server returning isError=True should produce a failed ToolResult."""
        require_env()
        from fastaiagent import MCPTool

        tool = MCPTool(
            name="echo_tool",
            server_url=stub_mcp_server,
            tool_name="echo_tool",
        )
        result = tool.execute({"message": "__error__"})
        assert not result.success, (
            "MCP isError=True did not produce a failed ToolResult"
        )
        assert result.error and "simulated tool error" in result.error

    def test_04_unreachable_server_raises(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent import MCPTool
        from fastaiagent._internal.errors import ToolExecutionError

        tool = MCPTool(
            name="dead_tool",
            server_url="http://127.0.0.1:1",  # reserved port, nothing listens
            tool_name="dead_tool",
        )
        with pytest.raises(ToolExecutionError):
            tool.execute({"message": "noop"})
