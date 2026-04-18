"""``fastaiagent mcp-serve`` — expose an Agent or Chain from a Python file as an MCP server."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import typer

mcp_app = typer.Typer(
    name="mcp",
    help="Expose an Agent or Chain as an MCP server.",
    no_args_is_help=True,
)


def _resolve_target(spec: str) -> Any:
    """Resolve ``path/to/module.py:attr`` or ``pkg.module:attr`` into a live object."""
    if ":" not in spec:
        raise typer.BadParameter(
            f"Expected 'path/to/file.py:attr' or 'pkg.module:attr', got {spec!r}"
        )
    module_part, attr = spec.rsplit(":", 1)
    path = Path(module_part)
    if path.exists():
        # File path — load the module by file name.
        module_name = path.stem
        spec_obj = importlib.util.spec_from_file_location(module_name, str(path))
        if spec_obj is None or spec_obj.loader is None:
            raise typer.BadParameter(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec_obj)
        # Make the parent directory importable so relative imports inside the
        # target module resolve.
        sys.path.insert(0, str(path.parent.resolve()))
        try:
            spec_obj.loader.exec_module(module)
        finally:
            pass
    else:
        # Dotted import path.
        module = importlib.import_module(module_part)

    if not hasattr(module, attr):
        raise typer.BadParameter(
            f"Module {module_part!r} has no attribute {attr!r}"
        )
    return getattr(module, attr)


@mcp_app.command("serve")
def serve(
    target: str = typer.Argument(
        ...,
        help=(
            "Target to expose: 'path/to/agent_file.py:my_agent' or "
            "'pkg.module:my_agent'. Must resolve to an Agent or Chain."
        ),
    ),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="MCP transport. Only 'stdio' is shipped in this release.",
    ),
    expose_tools: bool = typer.Option(
        False,
        "--expose-tools",
        help=(
            "If the target is an Agent, also expose each of its own tools as "
            "individual MCP tools (off by default — keeps the surface to a "
            "single primary tool)."
        ),
    ),
    name: str | None = typer.Option(
        None, "--name", help="Override the primary tool name."
    ),
) -> None:
    """Start an MCP server exposing the named Agent or Chain over stdio."""
    try:
        resolved = _resolve_target(target)
    except typer.BadParameter:
        raise
    except Exception as err:  # pragma: no cover - user import errors
        raise typer.BadParameter(f"Could not load {target!r}: {err}") from err

    # ``resolved`` should be an Agent, Chain, or already a FastAIAgentMCPServer.
    from fastaiagent.tool.mcp_server import FastAIAgentMCPServer

    if isinstance(resolved, FastAIAgentMCPServer):
        server = resolved
    elif hasattr(resolved, "as_mcp_server"):
        server = resolved.as_mcp_server(
            transport=transport, expose_tools=expose_tools, tool_name=name
        )
    else:
        raise typer.BadParameter(
            f"{target!r} resolved to {type(resolved).__name__} — expected "
            "Agent, Chain, or FastAIAgentMCPServer."
        )

    asyncio.run(server.run())
