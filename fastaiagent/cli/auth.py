"""``fastaiagent connect`` / ``disconnect`` / ``auth status`` — persist platform
credentials to ``~/.fastaiagent/credentials.toml`` and surface them to Python
code via environment variables.

The CLI writes the credentials file; Python code calls
``fa.connect(api_key=..., target=...)`` explicitly, or sources env vars the
CLI can print via ``auth env``. We deliberately do **not** have
``import fastaiagent`` auto-connect — that would be surprising. The CLI just
makes it easy to stash the credentials once and use them from anywhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

console = Console()

_DEFAULT_TARGET = "https://app.fastaiagent.net"
_CREDENTIALS_PATH = Path.home() / ".fastaiagent" / "credentials.toml"


def _read_credentials() -> dict[str, Any]:
    if not _CREDENTIALS_PATH.exists():
        return {}
    try:
        import tomllib  # type: ignore[import-untyped]  # py 3.11+, no stubs

        parsed: dict[str, Any] = tomllib.loads(_CREDENTIALS_PATH.read_text())
        return parsed
    except Exception as err:  # pragma: no cover — malformed TOML
        console.print(f"[yellow]Warning:[/yellow] could not parse {_CREDENTIALS_PATH}: {err}")
        return {}


def _write_credentials(data: dict[str, Any]) -> None:
    _CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# fastaiagent credentials — managed by `fastaiagent connect`. Do not share.", ""]
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, str):
            # crude but sufficient — we control the keys written here
            lines.append(f'{key} = "{value}"')
        else:
            lines.append(f"{key} = {value}")
    _CREDENTIALS_PATH.write_text("\n".join(lines) + "\n")
    _CREDENTIALS_PATH.chmod(0o600)


def _load_saved_credentials() -> dict[str, Any]:
    """Used by ``auth status`` and ``auth env`` to surface saved creds.

    Environment variables take precedence so CI runners can override.
    """
    saved = _read_credentials()
    merged = dict(saved)
    if os.environ.get("FASTAIAGENT_API_KEY"):
        merged["api_key"] = os.environ["FASTAIAGENT_API_KEY"]
        merged["_source"] = "env"
    elif saved:
        merged["_source"] = "file"
    if os.environ.get("FASTAIAGENT_TARGET"):
        merged["target"] = os.environ["FASTAIAGENT_TARGET"]
    if os.environ.get("FASTAIAGENT_PROJECT"):
        merged["project"] = os.environ["FASTAIAGENT_PROJECT"]
    return merged


def connect(
    api_key: str = typer.Option(
        ...,
        "--api-key",
        prompt=True,
        hide_input=True,
        help="Your FastAIAgent Platform API key.",
    ),
    target: str = typer.Option(
        _DEFAULT_TARGET,
        "--target",
        help="Platform URL.",
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        help="Optional project slug to scope trace uploads.",
    ),
) -> None:
    """Save Platform credentials to ~/.fastaiagent/credentials.toml and verify them.

    ``fa.connect(api_key=...)`` in Python is unchanged — this CLI just saves
    the key once so you don't have to pass it every time in scripts / CI.
    """
    from fastaiagent import client as _client

    try:
        _client.connect(api_key=api_key, target=target, project=project)
    except Exception as err:
        console.print(f"[red]Connection failed:[/red] {err}")
        raise typer.Exit(code=1) from err

    _write_credentials({"api_key": api_key, "target": target, "project": project})
    console.print(
        f"[green]✓[/green] Connected to [bold]{target}[/bold]. "
        f"Credentials saved to {_CREDENTIALS_PATH}."
    )


def disconnect() -> None:
    """Remove the saved credentials file and disconnect in-process."""
    from fastaiagent import client as _client

    _client.disconnect()
    if _CREDENTIALS_PATH.exists():
        _CREDENTIALS_PATH.unlink()
        console.print(f"[green]✓[/green] Removed {_CREDENTIALS_PATH}.")
    else:
        console.print("Not connected (no credentials file).")


auth_app = typer.Typer(
    name="auth",
    help="Show or inspect saved Platform credentials.",
    no_args_is_help=True,
)


@auth_app.command("status")
def auth_status() -> None:
    """Show whether credentials are saved and which source is active."""
    creds = _load_saved_credentials()
    if not creds.get("api_key"):
        console.print("[yellow]Not connected.[/yellow] Run `fastaiagent connect --api-key ...`.")
        raise typer.Exit(code=1)
    source = creds.get("_source", "file")
    target = creds.get("target", _DEFAULT_TARGET)
    project = creds.get("project")
    masked = creds["api_key"][:6] + "…" + creds["api_key"][-4:]
    console.print(f"[green]Connected[/green] (source: {source})")
    console.print(f"  Target:  {target}")
    console.print(f"  Project: {project or '(default)'}")
    console.print(f"  API key: {masked}")


@auth_app.command("env")
def auth_env() -> None:
    """Print saved credentials as shell `export` lines for sourcing.

    Usage::

        eval "$(fastaiagent auth env)"
    """
    creds = _load_saved_credentials()
    if not creds.get("api_key"):
        print("# Not connected. Run `fastaiagent connect --api-key ...` first.", file=sys.stderr)
        raise typer.Exit(code=1)
    print(f'export FASTAIAGENT_API_KEY="{creds["api_key"]}"')
    if creds.get("target"):
        print(f'export FASTAIAGENT_TARGET="{creds["target"]}"')
    if creds.get("project"):
        print(f'export FASTAIAGENT_PROJECT="{creds["project"]}"')
