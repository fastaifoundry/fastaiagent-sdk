"""``fastaiagent learn`` CLI subcommand.

Reads recent traces from ``local.db``, extracts durable facts via
:func:`fastaiagent.learn.extractor.run_extraction`, and persists them to
the ``learned_memory`` table for re-injection by
:class:`fastaiagent.agent.memory_blocks.PersistentFactBlock`.

Usage::

    # Default: agent-scope only, last 24h, no PII risk.
    fastaiagent learn --scope-id my-agent

    # Extract user-level facts (requires explicit opt-in).
    fastaiagent learn --scope user --scope-id user-42 --allow-personal

    # Preview without writing.
    fastaiagent learn --scope-id my-agent --dry-run

    # List currently-active facts.
    fastaiagent learn list

The ``--allow-personal`` flag is required for ``user`` and ``project``
scopes to avoid surprise PII extraction.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from fastaiagent.learn.extractor import run_extraction
from fastaiagent.learn.store import MemoryStore
from fastaiagent.llm.client import LLMClient

learn_app = typer.Typer(
    name="learn",
    help=(
        "Extract durable facts from past traces and re-inject them into "
        "future agent runs via PersistentFactBlock."
    ),
    no_args_is_help=False,
)
console = Console()


@learn_app.callback(invoke_without_command=True)
def extract(
    ctx: typer.Context,
    scope: str = typer.Option(
        "agent",
        "--scope",
        help="Fact scope: 'user' | 'project' | 'agent' (default: agent).",
    ),
    scope_id: str = typer.Option(
        "",
        "--scope-id",
        help="Scope identifier (agent name, user id, project id).",
    ),
    project_id: str = typer.Option(
        "",
        "--project-id",
        help="Project to scope DB queries to (matches the v4 project-scoping model).",
    ),
    last_hours: int = typer.Option(
        24,
        "--window",
        "--last-hours",
        help="Window of trace history to process, in hours.",
    ),
    max_facts: int = typer.Option(
        10,
        "--max-facts",
        help="Max facts extracted per trace.",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        help="LLM used for extraction (cheap+fast recommended).",
    ),
    provider: str = typer.Option(
        "openai", "--provider", help="LLM provider for extraction."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview candidates without writing to learned_memory.",
    ),
    allow_personal: bool = typer.Option(
        False,
        "--allow-personal",
        help=(
            "Required to extract user/project-scoped facts. Default extraction "
            "is agent-scope-only to avoid surprise PII."
        ),
    ),
) -> None:
    """Run the extraction loop over the configured trace window.

    If invoked without a subcommand, this is the action. Subcommands like
    ``list`` are routed via Typer if specified.
    """
    if ctx.invoked_subcommand is not None:
        # User asked for a subcommand; don't run extraction.
        return

    if scope not in ("user", "project", "agent"):
        raise typer.BadParameter(f"--scope must be user|project|agent, got {scope!r}")

    if scope in ("user", "project") and not allow_personal:
        console.print(
            "[red]Refusing to extract user/project-scoped facts without --allow-personal."
            "\nThis flag exists so PII extraction is always an explicit opt-in.[/red]"
        )
        raise typer.Exit(code=2)

    llm = LLMClient(provider=provider, model=model)
    store = MemoryStore()

    if dry_run:
        console.print("[yellow](dry-run — nothing will be written)[/yellow]")

    results = run_extraction(
        llm=llm,
        store=store,
        scope=scope,  # type: ignore[arg-type]  # validated above
        scope_id=scope_id,
        project_id=project_id,
        last_hours=last_hours,
        max_facts_per_trace=max_facts,
        dry_run=dry_run,
    )

    total_candidates = sum(len(r.candidates) for r in results)
    total_written = sum(len(r.written_ids) for r in results)
    error_count = sum(1 for r in results if r.error)

    table = Table(title=f"Extraction over {len(results)} traces")
    table.add_column("trace_id", style="dim")
    table.add_column("candidates", justify="right")
    table.add_column("written", justify="right")
    table.add_column("status")
    for r in results:
        status = "[red]error[/red]" if r.error else "ok"
        table.add_row(
            r.trace_id[:12] + "…",
            str(len(r.candidates)),
            str(len(r.written_ids)) if not dry_run else "—",
            status,
        )
    console.print(table)
    console.print(
        f"\n[bold]Total:[/bold] {total_candidates} candidate facts across "
        f"{len(results)} traces"
        + (f"  ([red]{error_count} errored[/red])" if error_count else "")
        + (f"  →  [green]{total_written} written[/green]" if not dry_run else "")
    )

    if dry_run and total_candidates > 0:
        console.print("\n[bold]Sample candidates:[/bold]")
        for r in results[:5]:
            for f in r.candidates[:3]:
                console.print(f"  - [{f.scope}/{f.scope_id or '*'}] {f.fact}")


@learn_app.command("list")
def list_facts(
    scope: str = typer.Option("agent", "--scope"),
    scope_id: str = typer.Option("", "--scope-id"),
    project_id: str = typer.Option("", "--project-id"),
    limit: int = typer.Option(50, "--limit"),
    show_superseded: bool = typer.Option(
        False, "--show-superseded", help="Include facts that have been replaced."
    ),
) -> None:
    """List currently-active learned facts (or include superseded with a flag)."""
    if scope not in ("user", "project", "agent"):
        raise typer.BadParameter(f"--scope must be user|project|agent, got {scope!r}")
    store = MemoryStore()
    if show_superseded:
        facts = store.list_all(project_id=project_id, limit=limit)
        facts = [f for f in facts if f.scope == scope and (not scope_id or f.scope_id == scope_id)]
    else:
        facts = store.list_active(
            scope=scope,  # type: ignore[arg-type]
            scope_id=scope_id,
            project_id=project_id,
            limit=limit,
        )

    if not facts:
        console.print(
            f"No {'(any)' if show_superseded else 'active'} facts for scope={scope!r}"
            + (f", scope_id={scope_id!r}" if scope_id else "")
        )
        return

    table = Table(title=f"{len(facts)} learned facts")
    table.add_column("id", style="dim")
    table.add_column("scope")
    table.add_column("scope_id")
    table.add_column("fact")
    table.add_column("source", style="dim")
    table.add_column("status")
    for f in facts:
        status = (
            f"[yellow]superseded by {f.superseded_by}[/yellow]"
            if f.superseded_by
            else "active"
        )
        table.add_row(
            str(f.id),
            f.scope,
            f.scope_id or "—",
            f.fact[:120],
            (f.source_trace_id or "")[:12] + ("…" if f.source_trace_id else "—"),
            status,
        )
    console.print(table)


@learn_app.command("supersede")
def supersede_cmd(
    old_id: int = typer.Argument(..., help="Existing fact id to mark as superseded."),
    new_id: int = typer.Argument(..., help="New fact id that replaces it."),
) -> None:
    """Mark one fact as superseded by another (manual conflict resolution)."""
    store = MemoryStore()
    store.supersede(old_id, new_id)
    console.print(f"[green]ok[/green] {old_id} superseded by {new_id}")


# Aliases the user can rely on if they prefer ``fastaiagent learn extract``.
@learn_app.command("extract", hidden=True)
def extract_alias(
    ctx: typer.Context,
    **kwargs,
) -> None:
    """Hidden alias — the default action of ``fastaiagent learn`` is extract."""
    # Typer doesn't easily forward all params, so users hitting this path
    # should just run ``fastaiagent learn`` directly.
    console.print(
        "Run [bold]fastaiagent learn[/bold] (without 'extract') — extract is the default."
    )
    raise typer.Exit(code=0)


# Allow ``python -m fastaiagent.learn.cli`` for ad-hoc invocation in tests.
if __name__ == "__main__":  # pragma: no cover
    learn_app()
