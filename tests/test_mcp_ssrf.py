"""security_audit_2 N15 — MCPTool egress goes through the SSRF guard.

MCPTool used to POST to ``server_url`` with a raw httpx client, so a
deserialized / attacker-influenced URL could reach cloud metadata or an
internal service. It now routes through ``asafe_http_request`` with
``allow_loopback=True`` (local MCP servers stay usable; metadata/RFC1918 do
not). These tests need no network: blocked hosts are rejected before any
socket is opened.
"""

from __future__ import annotations

import pytest

from fastaiagent._internal.errors import ToolExecutionError
from fastaiagent.tool import MCPTool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_url",
    [
        "http://169.254.169.254/latest/meta-data/",  # AWS metadata (link-local)
        "http://192.168.1.10:3000",  # RFC1918 private
        "http://10.0.0.5:3000",  # RFC1918 private
    ],
)
async def test_mcp_refuses_ssrf_targets(server_url: str) -> None:
    tool = MCPTool(name="t", server_url=server_url, tool_name="x")
    with pytest.raises(ToolExecutionError):
        await tool.aexecute({})


@pytest.mark.asyncio
async def test_mcp_allows_loopback_passes_guard() -> None:
    # Loopback is permitted for MCP, so the guard does NOT reject it; the call
    # then fails only because nothing is listening — a *connection* error, not
    # a refusal. Either way it must not be a "not a public address" refusal.
    tool = MCPTool(name="t", server_url="http://127.0.0.1:59999", tool_name="x")
    with pytest.raises(ToolExecutionError) as exc:
        await tool.aexecute({})
    assert "not a public" not in str(exc.value).lower()
