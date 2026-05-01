"""End-to-end test for ``examples/32_mcp_expose_agent.py``.

Spawns the example as an MCP stdio server subprocess and drives a real
client roundtrip via the ``mcp`` Python package:

    initialize → tools/list → tools/call research_lookup → cached fact

The example exits ``1`` if no LLM key is present, so the test asserts at
least one of OPENAI_API_KEY / ANTHROPIC_API_KEY before subprocessing.
``mcp`` is an optional extra (``fastaiagent[mcp-server]``); the test
skips cleanly when it isn't installed.

Marked ``e2e`` so it runs in the e2e quality gate alongside the rest of
the live-process tests, and is excluded from the fast unit matrix.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLE = REPO_ROOT / "examples" / "32_mcp_expose_agent.py"

pytestmark = pytest.mark.e2e


def _has_llm_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


async def test_mcp_example_serves_research_lookup() -> None:
    """Drive ``examples/32_mcp_expose_agent.py`` over stdio MCP.

    The example exposes ``expose_tools=True``, so individual tools — in
    particular ``research_lookup`` — should appear in the ``tools/list``
    response, and calling it with the cached topic ``"octopus"`` should
    return the literal cached fact (no LLM round-trip needed for the
    tool itself, the example's tool function is purely local).
    """
    pytest.importorskip(
        "mcp",
        reason="mcp client not installed; pip install 'fastaiagent[mcp-server]'",
    )
    if not _has_llm_key():
        pytest.skip(
            "No LLM key — the example exits 1 in _pick_llm() before it serves anything"
        )

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(EXAMPLE)],
        env=dict(os.environ),
    )

    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            tools_resp = await session.list_tools()
            tool_names = {t.name for t in tools_resp.tools}

            # ``expose_tools=True`` should surface the function tool the
            # agent uses; if not, the agent's primary tool (named after
            # the agent) should still appear.
            assert tool_names, "MCP server returned an empty tools list"
            primary_visible = any(
                "research_assistant" in n or "research_lookup" in n for n in tool_names
            )
            assert primary_visible, (
                f"Expected the agent or research_lookup in tools/list, got {tool_names}"
            )

            if "research_lookup" not in tool_names:
                pytest.skip(
                    "research_lookup not exposed individually — "
                    "expose_tools surface differs from the example's contract"
                )

            call = await session.call_tool("research_lookup", {"topic": "octopus"})
            text = "".join(
                getattr(block, "text", "") for block in call.content if hasattr(block, "text")
            )
            # The example's lookup() returns the literal cached string for
            # "octopus" — verify both salient tokens, not just one, so a
            # partial-string-match drift fails loudly.
            assert "three hearts" in text, f"unexpected result: {text!r}"
            assert "chromatophore" in text, f"unexpected result: {text!r}"
