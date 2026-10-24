"""security_audit_2 N15 — MCPTool egress goes through the SSRF guard.

``server_url`` can arrive from a payload (the runner, a trace, ``from_dict``), so
cloud metadata and private networks are refused; loopback stays allowed for local
MCP servers. Since 1.92.0 the official ``mcp`` client makes the requests and the
guard is an httpx transport that checks every request, each redirect hop included.

Each refusal is matched by its message: without the ``mcp`` extra the call also
fails (with the install hint), and a bare ``ToolExecutionError`` check would pass
for the wrong reason.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mcp")

from fastaiagent._internal.errors import ToolExecutionError  # noqa: E402
from fastaiagent.tool import MCPTool  # noqa: E402


@pytest.mark.parametrize(
    "server_url",
    [
        "http://169.254.169.254/latest/meta-data/",  # AWS metadata (link-local)
        "http://192.168.1.10:3000/mcp",  # RFC1918 private
        "http://10.0.0.5:3000/mcp",  # RFC1918 private
    ],
)
async def test_mcp_refuses_ssrf_targets(server_url: str) -> None:
    tool = MCPTool(name="t", server_url=server_url, tool_name="x")
    with pytest.raises(ToolExecutionError, match="not a public address"):
        await tool.aexecute({})


async def test_a_redirect_to_a_private_address_is_refused(capture_server: Any) -> None:
    capture_server.set_response_sequence(
        [(307, "", "text/plain", {"Location": "http://169.254.169.254/mcp"})]
    )
    tool = MCPTool(name="t", server_url=f"{capture_server.url}/mcp", tool_name="x")
    # mcp < 1.30 follows it into our guard; mcp >= 1.30 follows only same-origin
    # redirects and refuses this one itself. Either way it is never reached.
    with pytest.raises(ToolExecutionError, match="not a public address|not followed"):
        await tool.aexecute({})


async def test_loopback_passes_the_guard(official_mcp_server: Any) -> None:
    tool = MCPTool(name="add", server_url=official_mcp_server.url)
    assert (await tool.aexecute({"a": 1, "b": 1})).output == "2"
