"""One-time migrator from legacy stores (traces.db, checkpoints.db, YAML prompts) to local.db.

Safe to invoke multiple times — each step checks for source data and writes with
``INSERT OR IGNORE`` semantics so re-runs are no-ops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.ui.db import init_local_db


@dataclass
class MigrationReport:
    legacy_trace_db: Path | None = None
    legacy_checkpoint_db: Path | None = None
    legacy_prompt_dir: Path | None = None
    spans_migrated: int = 0
    checkpoints_migrated: int = 0
    prompts_migrated: int = 0
    prompt_versions_migrated: int = 0
    fragments_migrated: int = 0
    aliases_migrated: int = 0
    notes: list[str] = field(default_factory=list)

    def nothing_to_do(self) -> bool:
        return (
            self.legacy_trace_db is None
            and self.legacy_checkpoint_db is None
            and self.legacy_prompt_dir is None
        )


def migrate_to_local_db(
    *,
    target_db: Path | str | None = None,
    legacy_trace_db: Path | str | None = None,
    legacy_checkpoint_db: Path | str | None = None,
    legacy_prompt_dir: Path | str | None = None,
    force: bool = False,
) -> MigrationReport:
    """Copy legacy storage into the unified ``local.db``.

    Defaults scan for the legacy paths the SDK used before 0.8:
    ``./.fastaiagent/traces.db``, ``./.fastaiagent/checkpoints.db``, ``./.prompts/``.

    **Each source is imported once** (1.79.0). ``fastaiagent ui`` calls this on
    every start, and a row the user had deleted or pruned used to be copied back
    from the legacy file each time. An imported source is recorded in
    ``legacy_imports`` and skipped afterwards; ``force=True`` imports it again.

    **Imported spans and checkpoints are stored as already sent** (``synced=1``),
    the rule the v11/v13 schema upgrades follow: connecting must not push a user's
    history to the plane. Before 1.79.0 they took the column default and the next
    connected run pushed them. Publish history deliberately with
    :meth:`fastaiagent.trace.storage.TraceData.publish`.
    """
    target = Path(target_db) if target_db else Path(get_config().local_db_path)
    report = MigrationReport()

    trace_path = _path_if_exists(legacy_trace_db, Path(".fastaiagent/traces.db"))
    checkpoint_path = _path_if_exists(legacy_checkpoint_db, Path(".fastaiagent/checkpoints.db"))
    prompt_dir = _path_if_exists(legacy_prompt_dir, Path(".prompts"))

    if trace_path and trace_path.resolve() == target.resolve():
        trace_path = None
    if checkpoint_path and checkpoint_path.resolve() == target.resolve():
        checkpoint_path = None

    if not (trace_path or checkpoint_path or prompt_dir):
        return report

    local_db = init_local_db(target)
    try:
        if not force:
            trace_path = _unless_imported(local_db, trace_path, report)
            checkpoint_path = _unless_imported(local_db, checkpoint_path, report)
            prompt_dir = _unless_imported(local_db, prompt_dir, report)

        report.legacy_trace_db = trace_path
        report.legacy_checkpoint_db = checkpoint_path
        report.legacy_prompt_dir = prompt_dir

        if trace_path is not None:
            report.spans_migrated = _copy_rows(
                trace_path,
                local_db,
                table="spans",
                columns=(
                    "span_id",
                    "trace_id",
                    "parent_span_id",
                    "name",
                    "start_time",
                    "end_time",
                    "status",
                    "attributes",
                    "events",
                ),
                already_sent=True,
            )
            _record_import(local_db, trace_path, "traces", report.spans_migrated)
        if checkpoint_path is not None:
            report.checkpoints_migrated = _copy_rows(
                checkpoint_path,
                local_db,
                table="checkpoints",
                columns=(
                    "id",
                    "chain_name",
                    "execution_id",
                    "node_id",
                    "node_index",
                    "status",
                    "state_snapshot",
                    "node_input",
                    "node_output",
                    "iteration",
                    "iteration_counters",
                    "created_at",
                ),
                already_sent=True,
            )
            _record_import(local_db, checkpoint_path, "checkpoints", report.checkpoints_migrated)
        if prompt_dir is not None:
            (
                report.prompts_migrated,
                report.prompt_versions_migrated,
                report.fragments_migrated,
                report.aliases_migrated,
            ) = _copy_yaml_prompts(prompt_dir, local_db)
            _record_import(local_db, prompt_dir, "prompts", report.prompt_versions_migrated)
    finally:
        local_db.close()

    return report


def _unless_imported(db: SQLiteHelper, source: Path | None, report: MigrationReport) -> Path | None:
    """``source``, or ``None`` when this local.db already imported it."""
    if source is None:
        return None
    row = db.fetchone(
        "SELECT imported_at FROM legacy_imports WHERE source = ?", (str(source.resolve()),)
    )
    if row is None:
        return source
    report.notes.append(
        f"{source} was imported on {row['imported_at']}; skipped "
        f"(`fastaiagent migrate --force` imports it again)."
    )
    return None


def _record_import(db: SQLiteHelper, source: Path, kind: str, row_count: int) -> None:
    from datetime import datetime, timezone

    db.execute(
        """INSERT OR REPLACE INTO legacy_imports (source, kind, imported_at, row_count)
           VALUES (?, ?, ?, ?)""",
        (str(source.resolve()), kind, datetime.now(tz=timezone.utc).isoformat(), row_count),
    )


def _path_if_exists(explicit: Path | str | None, default: Path) -> Path | None:
    candidate = Path(explicit) if explicit is not None else default
    return candidate if candidate.exists() else None


def _copy_rows(
    source_db_path: Path,
    target: SQLiteHelper,
    *,
    table: str,
    columns: tuple[str, ...],
    already_sent: bool = False,
) -> int:
    """Copy ``columns`` of ``table``; rows already in the target are left alone.

    ``already_sent`` stores the copies with ``synced=1`` so the platform
    exporters never treat imported history as un-pushed.
    """
    cols = ", ".join(columns)
    # The legacy table has no ``synced`` column: it is added on the insert side only.
    insert_cols = cols + (", synced" if already_sent else "")
    placeholders = ", ".join("?" * len(columns)) + (", 1" if already_sent else "")
    with SQLiteHelper(source_db_path) as src:
        existing = src.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        if existing is None:
            return 0
        rows = src.fetchall(f"SELECT {cols} FROM {table}")
    if not rows:
        return 0
    target.executemany(
        f"INSERT OR IGNORE INTO {table} ({insert_cols}) VALUES ({placeholders})",
        [tuple(row[c] for c in columns) for row in rows],
    )
    return len(rows)


def _copy_yaml_prompts(prompt_dir: Path, target: SQLiteHelper) -> tuple[int, int, int, int]:
    prompts = 0
    versions = 0
    fragments = 0
    aliases = 0
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc).isoformat()

    for file in sorted(prompt_dir.glob("*.json")):
        if file.name.startswith("_fragment_"):
            name = file.stem[len("_fragment_") :]
            content = json.loads(file.read_text()).get("content", "")
            existing = target.fetchone("SELECT name FROM prompt_fragments WHERE name = ?", (name,))
            if existing is None:
                target.execute(
                    """INSERT INTO prompt_fragments (name, content, created_at, updated_at)
                       VALUES (?, ?, ?, ?)""",
                    (name, content, now, now),
                )
                fragments += 1
            continue

        data = json.loads(file.read_text())
        slug = data.get("name", file.stem)
        latest = data.get("latest_version", 1)

        existing = target.fetchone("SELECT slug FROM prompts WHERE slug = ?", (slug,))
        if existing is None:
            target.execute(
                """INSERT INTO prompts (slug, latest_version, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (slug, str(latest), now, now),
            )
            prompts += 1

        for version_entry in data.get("versions", []):
            version = version_entry.get("version", 1)
            row = target.fetchone(
                "SELECT version FROM prompt_versions WHERE slug = ? AND version = ?",
                (slug, str(version)),
            )
            if row is not None:
                continue
            target.execute(
                """INSERT INTO prompt_versions
                   (slug, version, template, variables, fragments, metadata,
                    created_at, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    slug,
                    str(version),
                    version_entry.get("template", ""),
                    json.dumps(version_entry.get("variables", [])),
                    json.dumps([]),
                    json.dumps(version_entry.get("metadata", {})),
                    now,
                    "migrate",
                ),
            )
            versions += 1

        for alias, version in (data.get("aliases") or {}).items():
            row = target.fetchone(
                "SELECT alias FROM prompt_aliases WHERE slug = ? AND alias = ?",
                (slug, alias),
            )
            if row is not None:
                continue
            target.execute(
                """INSERT INTO prompt_aliases (slug, alias, version)
                   VALUES (?, ?, ?)""",
                (slug, alias, str(version)),
            )
            aliases += 1

    return prompts, versions, fragments, aliases
