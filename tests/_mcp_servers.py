"""Real MCP servers for tests, built with the official ``mcp`` library.

``OfficialMCPServer`` serves FastMCP over streamable HTTP in a background
thread, so tests see the calls it received. Run this file as a script for the
same tools over stdio. Nothing here is a stub: both speak the protocol through
the library every MCP host uses.

No ``from __future__ import annotations``: FastMCP reads the tools' annotations
at runtime, and ``Context`` is imported inside :func:`build`.
"""

import socket
import sys
import threading
import time
from typing import Any


def build(calls: list[dict[str, Any]]) -> Any:
    from mcp.server.fastmcp import Context, FastMCP

    server = FastMCP("fastaiagent-tests")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        calls.append({"tool": "add", "a": a, "b": b})
        return a + b

    @server.tool()
    def fail(reason: str) -> str:
        """Always fails."""
        calls.append({"tool": "fail", "reason": reason})
        raise ValueError(f"refused: {reason}")

    @server.tool()
    def notify(text: str) -> str:
        """Notify the customer."""
        calls.append({"tool": "notify", "text": text})
        return f"sent: {text}"

    @server.tool()
    def whoami(ctx: Context) -> str:
        """The Authorization header this request carried (HTTP only)."""
        request = getattr(ctx.request_context, "request", None)
        return request.headers.get("authorization", "<none>") if request else "<stdio>"

    return server


class OfficialMCPServer:
    """FastMCP on streamable HTTP at ``url``; ``calls`` lists every tool call."""

    def __init__(self) -> None:
        import uvicorn

        self.calls: list[dict[str, Any]] = []
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/mcp"
        app = build(self.calls).streamable_http_app()
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> "OfficialMCPServer":
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("the MCP test server did not start")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


#: ``MCPTool(command=STDIO_COMMAND)`` starts this file as a stdio server.
STDIO_COMMAND = [sys.executable, __file__]

if __name__ == "__main__":
    build([]).run(transport="stdio")
