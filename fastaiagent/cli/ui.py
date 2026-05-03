"""`fastaiagent ui` — start the local web UI or reset its password."""

from __future__ import annotations

import getpass
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console

logger = logging.getLogger(__name__)

ui_app = typer.Typer(
    help="Local web UI for traces, prompts, evals, and guardrails."
)
console = Console()


_EXTRAS_HINT = (
    "The UI requires optional dependencies. Install them with:\n\n"
    "    pip install 'fastaiagent[ui]'\n"
)


def _ensure_ui_extra() -> None:
    """Import-check the web framework; raise a friendly error if missing."""
    missing: list[str] = []
    for mod in ("fastapi", "uvicorn", "bcrypt", "itsdangerous"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        console.print(f"[red]Missing packages:[/red] {', '.join(missing)}")
        console.print(_EXTRAS_HINT)
        raise typer.Exit(code=2)


def _resolve_paths(
    db: Path | None, auth: Path | None
) -> tuple[Path, Path]:
    from fastaiagent._internal.config import get_config
    from fastaiagent.ui.auth import default_auth_path

    db_path = db or Path(get_config().local_db_path)
    auth_path = auth or default_auth_path()
    return db_path, auth_path


def _maybe_migrate(db_path: Path) -> None:
    """Auto-migrate legacy stores into local.db on first UI start."""
    from fastaiagent.ui.migration import migrate_to_local_db

    report = migrate_to_local_db(target_db=db_path)
    if not report.nothing_to_do():
        console.print("[dim]Migrating legacy storage → local.db…[/dim]")
        if report.legacy_trace_db:
            console.print(
                f"  traces: {report.spans_migrated} spans from "
                f"{report.legacy_trace_db}"
            )
        if report.legacy_checkpoint_db:
            console.print(
                f"  checkpoints: {report.checkpoints_migrated} from "
                f"{report.legacy_checkpoint_db}"
            )
        if report.legacy_prompt_dir:
            console.print(
                f"  prompts: {report.prompts_migrated} prompts, "
                f"{report.prompt_versions_migrated} versions from "
                f"{report.legacy_prompt_dir}"
            )


def _prompt_first_run(auth_path: Path) -> None:
    """Interactive username/password setup written to auth.json."""
    from fastaiagent.ui.auth import create_auth_file

    console.print("[bold]FastAIAgent Local UI — first run[/bold]")
    username = typer.prompt("Set a username", default="admin")
    while True:
        password = getpass.getpass("Set a password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password == confirm and password:
            break
        console.print(
            "[yellow]Passwords did not match or were empty — try again.[/yellow]"
        )
    create_auth_file(username, password, path=auth_path)
    console.print(f"[green]\N{CHECK MARK} Credentials saved to {auth_path}[/green]")


def _start_server(
    *, host: str, port: int, db_path: Path, auth_path: Path, no_auth: bool, no_open: bool
) -> None:
    import uvicorn

    from fastaiagent.ui.server import build_app

    app = build_app(
        db_path=str(db_path), auth_path=auth_path, no_auth=no_auth
    )
    url = f"http://{host}:{port}"
    console.print(f"[bold]Starting UI on {url}[/bold]")
    if not no_open:
        try:
            import webbrowser

            webbrowser.open_new_tab(url)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to open browser", exc_info=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")


@ui_app.callback(invoke_without_command=True)
def default(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7842, "--port"),
    no_auth: bool = typer.Option(False, "--no-auth", help="Skip local auth (throwaway use)."),
    no_open: bool = typer.Option(False, "--no-open"),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Override local.db path (defaults to FASTAIAGENT_LOCAL_DB).",
    ),
    auth: Path | None = typer.Option(
        None,
        "--auth-file",
        help="Override auth.json path.",
    ),
) -> None:
    """Default: start the server (equivalent to `fastaiagent ui start`)."""
    if ctx.invoked_subcommand is not None:
        return
    _run_start(
        host=host,
        port=port,
        no_auth=no_auth,
        no_open=no_open,
        db=db,
        auth=auth,
    )


@ui_app.command("start")
def start(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7842, "--port"),
    no_auth: bool = typer.Option(False, "--no-auth"),
    no_open: bool = typer.Option(False, "--no-open"),
    db: Path | None = typer.Option(None, "--db"),
    auth: Path | None = typer.Option(None, "--auth-file"),
) -> None:
    """Start the local web UI server."""
    _run_start(
        host=host,
        port=port,
        no_auth=no_auth,
        no_open=no_open,
        db=db,
        auth=auth,
    )


def _run_start(
    *,
    host: str,
    port: int,
    no_auth: bool,
    no_open: bool,
    db: Path | None,
    auth: Path | None,
) -> None:
    _ensure_ui_extra()
    from fastaiagent.ui.auth import auth_file_exists

    db_path, auth_path = _resolve_paths(db, auth)
    _maybe_migrate(db_path)
    if not no_auth and not auth_file_exists(auth_path):
        try:
            _prompt_first_run(auth_path)
        except (KeyboardInterrupt, EOFError):
            console.print("[red]Cancelled.[/red]")
            raise typer.Exit(code=1) from None
    _start_server(
        host=host,
        port=port,
        db_path=db_path,
        auth_path=auth_path,
        no_auth=no_auth,
        no_open=no_open,
    )


@ui_app.command("reset-password")
def reset_password(
    auth: Path | None = typer.Option(None, "--auth-file"),
) -> None:
    """Delete auth.json so the next `ui start` can prompt for new credentials."""
    from fastaiagent.ui.auth import default_auth_path, delete_auth_file

    target = auth or default_auth_path()
    if delete_auth_file(target):
        console.print(
            f"[green]Deleted {target}. Run `fastaiagent ui` to set new credentials.[/green]"
        )
    else:
        console.print(f"[dim]{target} did not exist; nothing to delete.[/dim]")
        sys.exit(0)
