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
) -> MigrationReport:
    """Copy legacy storage into the unified ``local.db``.

    Defaults scan for the legacy paths the SDK used before 0.8:
    ``./.fastaiagent/traces.db``, ``./.fastaiagent/checkpoints.db``, ``./.prompts/``.
    """
    target = Path(target_db) if target_db else Path(get_config().local_db_path)
    report = MigrationReport()

    trace_path = _path_if_exists(legacy_trace_db, Path(".fastaiagent/traces.db"))
    checkpoint_path = _path_if_exists(
        legacy_checkpoint_db, Path(".fastaiagent/checkpoints.db")
    )
    prompt_dir = _path_if_exists(legacy_prompt_dir, Path(".prompts"))

    if trace_path and trace_path.resolve() == target.resolve():
        trace_path = None
    if checkpoint_path and checkpoint_path.resolve() == target.resolve():
        checkpoint_path = None

    if not (trace_path or checkpoint_path or prompt_dir):
        return report

    report.legacy_trace_db = trace_path
    report.legacy_checkpoint_db = checkpoint_path
    report.legacy_prompt_dir = prompt_dir

    local_db = init_local_db(target)
    try:
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
            )
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
            )
        if prompt_dir is not None:
            (
                report.prompts_migrated,
                report.prompt_versions_migrated,
                report.fragments_migrated,
                report.aliases_migrated,
            ) = _copy_yaml_prompts(prompt_dir, local_db)
    finally:
        local_db.close()

    return report


def _path_if_exists(
    explicit: Path | str | None, default: Path
) -> Path | None:
    candidate = Path(explicit) if explicit is not None else default
    return candidate if candidate.exists() else None


def _copy_rows(
    source_db_path: Path,
    target: SQLiteHelper,
    *,
    table: str,
    columns: tuple[str, ...],
) -> int:
    cols = ", ".join(columns)
    placeholders = ", ".join("?" * len(columns))
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
        f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})",
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
            existing = target.fetchone(
                "SELECT name FROM prompt_fragments WHERE name = ?", (name,)
            )
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
