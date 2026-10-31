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
            "(fields: trace_id, original_output, new_output, tool_dispositions, …). "
            "If omitted, prints to stdout."
        ),
    ),
    mode: str = typer.Option(
        "live", "--mode", help="live | recorded | deterministic (how the model is called)."
    ),
    on_miss: str = typer.Option(
        "live", "--on-miss", help="recorded mode: live | error when the rerun asks more."
    ),
    tools: str | None = typer.Option(
        None,
        "--tools",
        help=(
            "block | error | live: what to do with tools you did not --stub or "
            "--allow-tool. Default: block, or FASTAIAGENT_REPLAY_TOOL_POLICY."
        ),
    ),
    allow_tool: list[str] = typer.Option(
        [], "--allow-tool", help="Run this tool live (repeatable). Required for REST/MCP tools."
    ),
    stub: list[str] = typer.Option(
        [], "--stub", help="NAME=JSON: this tool returns JSON instead of running (repeatable)."
    ),
    db: str | None = typer.Option(None, "--db", help="local.db holding the trace to fork."),
    rerun_db: str | None = typer.Option(
        None, "--rerun-db", help="Write the rerun's trace to this local.db."
    ),
    export: bool = typer.Option(
        False, "--export/--no-export", help="Export the rerun to the plane (off by default)."
    ),
) -> None:
    """Fork a past trace at a step, optionally modify it, and rerun it.

    This is the CLI surface for ``Replay.load(trace_id).fork_at(step)
    .modify_prompt(...).modify_input(...).rerun()``.

    Since 1.89.0 a rerun runs only the tools you give it: every other tool is
    blocked (``--tools block``, the default). Use ``--stub NAME=JSON`` or
    ``--allow-tool NAME`` per tool, or ``--tools live`` for the old behaviour.
    The rerun is not exported unless you pass ``--export``.
    """
    from fastaiagent._internal.errors import ReplayError
    from fastaiagent.trace.replay import Replay
    from fastaiagent.trace.storage import TraceStore

    stubs: dict[str, object] = {}
    for item in stub:
        name, sep, raw = item.partition("=")
        if not sep or not name:
            raise typer.BadParameter(f"--stub expects NAME=JSON, got {item!r}")
        try:
            stubs[name] = json.loads(raw)
        except json.JSONDecodeError:
            stubs[name] = raw  # a bare string is a fine canned result

    replay = Replay.load(trace_id, store=TraceStore(db_path=db) if db else None)
    try:
        forked = replay.fork_at(step)
    except ReplayError as exc:  # a step out of range, or a trace that is not an agent run
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    forked._surface = "cli"
    if prompt is not None:
        forked.modify_prompt(prompt)
    if input is not None:
        forked.modify_input({"input": input})
    try:
        forked.with_determinism(mode, on_miss=on_miss)  # type: ignore[arg-type]
        forked.with_tool_policy(tools or forked._tool_policy, allow=allow_tool)  # type: ignore[arg-type]
        for name, value in stubs.items():
            forked.with_tool_stub(name, value)
        if rerun_db:
            forked.with_trace_db(rerun_db)
        if export:
            forked.with_export(True)
        result = forked.rerun()
    except ReplayError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    payload = {
        "trace_id": result.trace_id,
        "original_output": result.original_output,
        "new_output": result.new_output,
        "steps_executed": result.steps_executed,
        "determinism": result.determinism,
        "tool_dispositions": result.tool_dispositions,
        "blocked_tool_calls": result.blocked_tool_calls,
        "exported": result.exported,
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
