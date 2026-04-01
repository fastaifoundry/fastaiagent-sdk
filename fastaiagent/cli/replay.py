"""CLI commands for Agent Replay."""

import typer
from rich.console import Console

replay_app = typer.Typer()
console = Console()


@replay_app.command("show")
def show_replay(trace_id: str = typer.Argument(..., help="Trace ID to replay")):
    """Show replay steps for a trace."""
    from fastaiagent.trace.replay import Replay

    replay = Replay.load(trace_id)
    console.print(replay.summary())


@replay_app.command("inspect")
def inspect_step(
    trace_id: str = typer.Argument(..., help="Trace ID"),
    step: int = typer.Argument(..., help="Step number to inspect"),
):
    """Inspect a specific replay step."""
    from fastaiagent.trace.replay import Replay

    replay = Replay.load(trace_id)
    step_data = replay.inspect(step)
    console.print(f"[bold]Step {step_data.step}:[/bold] {step_data.span_name}")
    console.print(f"  Timestamp: {step_data.timestamp}")
    console.print(f"  Attributes: {step_data.attributes}")
