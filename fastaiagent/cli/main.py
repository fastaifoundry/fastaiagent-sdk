"""FastAIAgent CLI entry point."""

import typer

from fastaiagent.cli.eval import eval_app
from fastaiagent.cli.kb import kb_app
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


@app.command()
def version():
    """Show the SDK version."""
    from fastaiagent._version import __version__

    typer.echo(f"fastaiagent {__version__}")


@app.command()
def push(
    api_key: str = typer.Option(..., envvar="FASTAIAGENT_API_KEY", help="Platform API key"),
    target: str = typer.Option("https://app.fastaiagent.net", envvar="FASTAIAGENT_TARGET", help="Platform URL"),
    agent: str | None = typer.Option(None, help="Agent module:name to push (e.g., myapp:support_agent)"),
    chain: str | None = typer.Option(None, help="Chain module:name to push"),
):
    """Push resources to the FastAIAgent platform."""
    from rich.console import Console

    console = Console()

    if not agent and not chain:
        console.print("[red]Specify --agent or --chain to push.[/red]")
        raise typer.Exit(1)

    from fastaiagent.client import FastAI

    fa = FastAI(api_key=api_key, target=target)
    console.print(f"Connecting to {target}...")

    if agent:
        console.print(f"[dim]Push via Python API: fa.push(my_agent)[/dim]")
    if chain:
        console.print(f"[dim]Push via Python API: fa.push(my_chain)[/dim]")


if __name__ == "__main__":
    app()
