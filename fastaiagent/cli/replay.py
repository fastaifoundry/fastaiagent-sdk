"""CLI commands for Agent Replay."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

replay_app = typer.Typer()
console = Console()


@replay_app.command("fork")
def fork_and_rerun(
    trace_id: str = typer.Argument(..., help="Trace ID to fork."),
    step: int = typer.Option(
        0,
        "--step",
        help="Step to fork at. 0 = from the top with modifications applied.",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Override the agent's system prompt before re-running.",
    ),
    input: str | None = typer.Option(  # noqa: A002 - matches replay API kw
        None,
        "--input",
        help="Override the original user input before re-running.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        help=(
            "Write the rerun result to this JSON file "
            "(fields: trace_id, original_output, new_output). "
            "If omitted, prints to stdout."
        ),
    ),
) -> None:
    """Fork a past trace at a step, optionally modify it, and rerun it.

    This is the CLI surface for ``Replay.load(trace_id).fork_at(step)
    .modify_prompt(...).modify_input(...).rerun()``.
    """
    from fastaiagent.trace.replay import Replay

    replay = Replay.load(trace_id)
    forked = replay.fork_at(step)
    if prompt is not None:
        forked.modify_prompt(prompt)
    if input is not None:
        forked.modify_input({"input": input})
    result = forked.rerun()

    payload = {
        "trace_id": result.trace_id,
        "original_output": result.original_output,
        "new_output": result.new_output,
        "steps_executed": result.steps_executed,
    }
    if output is None:
        console.print(payload)
    else:
        Path(output).write_text(json.dumps(payload, indent=2))
        console.print(f"[green]✓[/green] Wrote rerun result to {output}")


@replay_app.command("show")
def show_replay(trace_id: str = typer.Argument(..., help="Trace ID to replay")) -> None:
    """Show replay steps for a trace."""
    from fastaiagent.trace.replay import Replay

    replay = Replay.load(trace_id)
    console.print(replay.summary())


@replay_app.command("inspect")
def inspect_step(
    trace_id: str = typer.Argument(..., help="Trace ID"),
    step: int = typer.Argument(..., help="Step number to inspect"),
) -> None:
    """Inspect a specific replay step."""
    from fastaiagent.trace.replay import Replay

    replay = Replay.load(trace_id)
    step_data = replay.inspect(step)
    console.print(f"[bold]Step {step_data.step}:[/bold] {step_data.span_name}")
    console.print(f"  Timestamp: {step_data.timestamp}")
    console.print(f"  Attributes: {step_data.attributes}")
