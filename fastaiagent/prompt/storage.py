"""SQLite-backed local prompt storage, targeting the unified ``local.db``.

Replaces the previous YAML-per-file layout. Prompts, versions, aliases, and
fragments live in the same SQLite file that backs traces, checkpoints, and
eval runs — one file per project.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastaiagent._internal.errors import FragmentNotFoundError, PromptNotFoundError
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.prompt.fragment import Fragment
from fastaiagent.prompt.prompt import Prompt
from fastaiagent.ui.db import init_local_db


def _resolve_db_file(path: str | Path) -> Path:
    """Accept either a DB file path or a directory (legacy behavior).

    If ``path`` has a ``.db`` suffix, it's used as-is. Otherwise ``path`` is
    treated as a directory and ``local.db`` is placed inside it. This keeps
    ``PromptRegistry(path=".prompts/")`` working for tests and CLI flags
    that still pass a directory.
    """
    p = Path(path)
    if p.suffix == ".db":
        return p
    return p / "local.db"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class SQLiteStorage:
    """Drop-in replacement for :class:`YAMLStorage` backed by local.db."""

    def __init__(self, path: str | Path):
        self.file = _resolve_db_file(path)
        self._db: SQLiteHelper = init_local_db(self.file)

    # --- Prompts ---------------------------------------------------------

    def save_prompt(
        self, prompt: Prompt, *, project_id: str | None = None
    ) -> None:
        """Persist a prompt + its version row.

        ``project_id`` overrides the auto-detected project id when given.
        Pass this from contexts (like the Local UI) where the *active*
        project is known explicitly — falling back to
        ``safe_get_project_id()`` would stamp the row with the cwd-derived
        name, which can diverge from the UI's scope when the same DB is
        served under a different project.
        """
        from fastaiagent._internal.project import safe_get_project_id

        pid = project_id if project_id is not None else safe_get_project_id()
        now = _now_iso()
        existing = self._db.fetchone(
            "SELECT created_at FROM prompts WHERE slug = ?",
            (prompt.name,),
        )
        if existing is None:
            self._db.execute(
                """INSERT INTO prompts (slug, latest_version, created_at, updated_at, project_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (prompt.name, str(prompt.version), now, now, pid),
            )
        else:
            self._db.execute(
                """UPDATE prompts
                   SET latest_version = ?, updated_at = ?
                   WHERE slug = ?""",
                (str(prompt.version), now, prompt.name),
            )
        self._db.execute(
            """INSERT OR REPLACE INTO prompt_versions
               (slug, version, template, variables, fragments, metadata,
                created_at, created_by, project_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prompt.name,
                str(prompt.version),
                prompt.template,
                json.dumps(prompt.variables),
                json.dumps([]),
                json.dumps(prompt.metadata),
                now,
                "code",
                pid,
            ),
        )

    def load_prompt(
        self, name: str, version: int | None = None, alias: str | None = None
    ) -> Prompt:
        prompt_row = self._db.fetchone(
            "SELECT * FROM prompts WHERE slug = ?", (name,)
        )
        if prompt_row is None:
            raise PromptNotFoundError(f"Prompt '{name}' not found")

        resolved_version = version
        if alias is not None:
            alias_row = self._db.fetchone(
                "SELECT version FROM prompt_aliases WHERE slug = ? AND alias = ?",
                (name, alias),
            )
            if alias_row is None:
                raise PromptNotFoundError(
                    f"Alias '{alias}' not found for prompt '{name}'"
                )
            resolved_version = int(alias_row["version"])

        if resolved_version is not None:
            row = self._db.fetchone(
                "SELECT * FROM prompt_versions WHERE slug = ? AND version = ?",
                (name, str(resolved_version)),
            )
            if row is None:
                raise PromptNotFoundError(
                    f"Version {resolved_version} not found for prompt '{name}'"
                )
            return self._row_to_prompt(row)

        latest_row = self._db.fetchone(
            """SELECT * FROM prompt_versions
               WHERE slug = ?
               ORDER BY CAST(version AS INTEGER) DESC
               LIMIT 1""",
            (name,),
        )
        if latest_row is None:
            raise PromptNotFoundError(f"Prompt '{name}' has no versions")
        return self._row_to_prompt(latest_row)

    def delete_prompt(
        self, name: str, *, project_id: str | None = None
    ) -> int:
        """Delete a prompt and every row that references it.

        Removes ``prompts``, ``prompt_versions``, and ``prompt_aliases`` rows
        for ``name``. When ``project_id`` is given, only rows in that project
        are deleted — multi-project DBs stay intact for siblings.

        Returns the number of version rows that were removed (0 means the
        prompt didn't exist in this scope). Trace rows that referenced the
        prompt are left alone — historical traces should keep showing the
        prompt name they ran with even after the live definition is gone.
        """
        params: tuple
        if project_id is None:
            params = (name,)
            scope_extra = ""
        else:
            params = (name, project_id)
            scope_extra = " AND project_id = ?"

        # Capture the version count first so the caller can tell whether
        # anything was actually deleted.
        count_row = self._db.fetchone(
            f"SELECT COUNT(*) AS n FROM prompt_versions "
            f"WHERE slug = ?{scope_extra}",
            params,
        )
        deleted = int((count_row or {}).get("n") or 0)
        if deleted == 0:
            # Nothing here under this scope. Caller turns this into a 404.
            return 0

        # ``prompt_aliases`` doesn't carry a ``project_id`` column (it
        # joins on slug+alias, which is already globally unique). Drop the
        # scope filter for it but keep it for the project-scoped tables.
        for table, t_params in (
            ("prompt_versions", params),
            ("prompts", params),
            ("prompt_aliases", (name,)),
        ):
            scope = scope_extra if table != "prompt_aliases" else ""
            self._db.execute(
                f"DELETE FROM {table} WHERE slug = ?{scope}",
                t_params,
            )
        return deleted

    def set_alias(self, name: str, version: int, alias: str) -> None:
        prompt_row = self._db.fetchone(
            "SELECT slug FROM prompts WHERE slug = ?", (name,)
        )
        if prompt_row is None:
            raise PromptNotFoundError(f"Prompt '{name}' not found")
        self._db.execute(
            """INSERT OR REPLACE INTO prompt_aliases (slug, alias, version)
               VALUES (?, ?, ?)""",
            (name, alias, str(version)),
        )

    def list_prompts(self) -> list[dict[str, Any]]:
        rows = self._db.fetchall(
            """SELECT p.slug AS name,
                      p.latest_version AS latest_version,
                      COUNT(v.version) AS version_count
               FROM prompts p
               LEFT JOIN prompt_versions v ON v.slug = p.slug
               GROUP BY p.slug
               ORDER BY p.slug"""
        )
        results: list[dict[str, Any]] = []
        for row in rows:
            latest = row["latest_version"]
            try:
                latest_int: int | str = int(latest) if latest is not None else 1
            except (TypeError, ValueError):
                latest_int = latest
            results.append(
                {
                    "name": row["name"],
                    "latest_version": latest_int,
                    "versions": row["version_count"],
                }
            )
        return results

    # --- Fragments -------------------------------------------------------

    def save_fragment(self, fragment: Fragment) -> None:
        now = _now_iso()
        self._db.execute(
            """INSERT INTO prompt_fragments (name, content, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                 content = excluded.content,
                 updated_at = excluded.updated_at""",
            (fragment.name, fragment.content, now, now),
        )

    def load_fragment(self, name: str) -> Fragment:
        row = self._db.fetchone(
            "SELECT name, content FROM prompt_fragments WHERE name = ?",
            (name,),
        )
        if row is None:
            raise FragmentNotFoundError(f"Fragment '{name}' not found")
        return Fragment(name=row["name"], content=row["content"])

    # --- Internal --------------------------------------------------------

    def _row_to_prompt(self, row: dict[str, Any]) -> Prompt:
        try:
            version_int = int(row["version"])
        except (TypeError, ValueError):
            version_int = 1
        variables = json.loads(row["variables"]) if row.get("variables") else []
        metadata = json.loads(row["metadata"]) if row.get("metadata") else {}
        return Prompt(
            name=row["slug"],
            template=row["template"] or "",
            variables=variables,
            version=version_int,
            metadata=metadata,
        )
