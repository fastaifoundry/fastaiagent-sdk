"""``fastaiagent setup-checkpointer`` — CLI to provision a durability backend.

For SQLite the schema is auto-created on first ``Chain.aexecute`` /
``Agent.arun``, so this command is mostly for verification: it opens the
database, runs migrations idempotently, and prints a summary. For Postgres
the same idempotent migration runs against the configured schema —
useful as a deploy step before the first production checkpoint write.
"""

from __future__ import annotations

import typer


def _setup_command(
    backend: str = typer.Option(
        "sqlite",
        "--backend",
        case_sensitive=False,
        help="Durability backend: 'sqlite' (default) or 'postgres'.",
    ),
    connection_string: str | None = typer.Option(
        None,
        "--connection-string",
        help=(
            "For sqlite: path to the database file. For postgres: a "
            "PostgreSQL DSN, e.g. postgresql://user:pass@host/db. "
            "If omitted for sqlite, the configured local_db_path is used."
        ),
    ),
    schema: str = typer.Option(
        "fastaiagent",
        "--schema",
        help="Postgres schema name (postgres backend only).",
    ),
) -> None:
    """Run idempotent migrations against the configured backend."""
    backend_norm = backend.strip().lower()
    if backend_norm == "sqlite":
        from fastaiagent.checkpointers import SQLiteCheckpointer

        store = SQLiteCheckpointer(db_path=connection_string)
        store.setup()
        try:
            count = len(store.list("__nonexistent__"))
        except Exception:
            count = 0
        store.close()
        typer.secho(
            f"SQLite checkpointer ready at {store.db_path!r}.",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"  initial checkpoint count: {count}")
        return

    if backend_norm == "postgres":
        if not connection_string:
            raise typer.BadParameter(
                "--connection-string is required for the postgres backend "
                "(e.g. postgresql://user:pass@host/db)."
            )
        try:
            from fastaiagent.checkpointers.postgres import PostgresCheckpointer
        except ImportError as e:
            raise typer.BadParameter(
                "Install the postgres extra: `pip install 'fastaiagent[postgres]'`."
            ) from e

        pg_store = PostgresCheckpointer(connection_string, schema=schema)
        pg_store.setup()
        pg_store.close()
        typer.secho(
            f"Postgres checkpointer ready at schema {schema!r}.",
            fg=typer.colors.GREEN,
        )
        return

    raise typer.BadParameter(f"Unknown --backend {backend!r}. Use 'sqlite' or 'postgres'.")


def register(app: typer.Typer) -> None:
    """Attach ``setup-checkpointer`` to the parent CLI."""
    app.command(
        name="setup-checkpointer",
        help="Provision the durability backend (SQLite or Postgres).",
    )(_setup_command)


__all__ = ["register"]
