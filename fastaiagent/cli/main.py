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
def version() -> None:
    """Show the SDK version."""
    from fastaiagent._version import __version__

    typer.echo(f"fastaiagent {__version__}")


if __name__ == "__main__":
    app()
