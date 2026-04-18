"""FastAIAgent CLI entry point."""

from __future__ import annotations

import importlib.util

import typer

from fastaiagent.cli.agent import agent_app
from fastaiagent.cli.auth import auth_app, connect, disconnect
from fastaiagent.cli.eval import eval_app
from fastaiagent.cli.kb import kb_app
from fastaiagent.cli.mcp import mcp_app
from fastaiagent.cli.prompts import prompts_app
from fastaiagent.cli.replay import replay_app
from fastaiagent.cli.traces import traces_app

app = typer.Typer(
    name="fastaiagent",
    help="FastAIAgent SDK — Build, debug, evaluate, and operate AI agents.",
    no_args_is_help=True,
)

app.add_typer(traces_app, name="traces", help="Manage traces")
app.add_typer(replay_app, name="replay", help="Agent Replay")
app.add_typer(eval_app, name="eval", help="Run evaluations")
app.add_typer(prompts_app, name="prompts", help="Manage prompts")
app.add_typer(kb_app, name="kb", help="Manage knowledge bases")
app.add_typer(mcp_app, name="mcp", help="Expose an Agent or Chain as an MCP server")
app.add_typer(agent_app, name="agent", help="Run an Agent or Chain as a service")
app.add_typer(auth_app, name="auth", help="Inspect saved Platform credentials")

# Top-level connect/disconnect for the common case.
app.command(name="connect", help="Save Platform credentials and verify the key.")(connect)
app.command(name="disconnect", help="Remove saved Platform credentials.")(disconnect)


# Known optional extras and the packages that identify them. We probe each
# to produce an honest "what's installed" readout in `fastaiagent version`.
_KNOWN_EXTRAS: list[tuple[str, str]] = [
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    ("langchain", "langchain_core"),
    ("crewai", "crewai"),
    ("kb", "faiss"),
    ("qdrant", "qdrant_client"),
    ("chroma", "chromadb"),
    ("mcp-server", "mcp"),
    ("otel-export", "opentelemetry.exporter.otlp"),
]


def _installed_extras() -> list[str]:
    names: list[str] = []
    for extra_name, module_name in _KNOWN_EXTRAS:
        if importlib.util.find_spec(module_name) is not None:
            names.append(extra_name)
    return names


@app.command()
def version() -> None:
    """Show the SDK version and which optional extras are installed."""
    from fastaiagent._version import __version__

    extras = _installed_extras()
    if extras:
        typer.echo(f"fastaiagent {__version__} [{', '.join(extras)}]")
    else:
        typer.echo(f"fastaiagent {__version__}")


if __name__ == "__main__":
    app()
