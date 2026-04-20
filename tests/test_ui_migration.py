"""Tests for the legacy -> local.db migrator.

Uses real SQLite files and real YAML-like JSON prompt files — no mocks.
"""

from __future__ import annotations

import json

from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.ui.migration import migrate_to_local_db


def _seed_legacy_traces(path, rows):
    with SQLiteHelper(path) as db:
        db.execute(
            """CREATE TABLE spans (
                span_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                parent_span_id TEXT,
                name TEXT,
                start_time TEXT,
                end_time TEXT,
                status TEXT DEFAULT 'OK',
                attributes TEXT DEFAULT '{}',
                events TEXT DEFAULT '[]'
            )"""
        )
        for r in rows:
            db.execute(
                "INSERT INTO spans (span_id, trace_id, name) VALUES (?, ?, ?)",
                r,
            )


def _seed_legacy_checkpoints(path, rows):
    with SQLiteHelper(path) as db:
        db.execute(
            """CREATE TABLE checkpoints (
                id TEXT PRIMARY KEY,
                chain_name TEXT NOT NULL,
                execution_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                node_index INTEGER,
                status TEXT DEFAULT 'completed',
                state_snapshot TEXT DEFAULT '{}',
                node_input TEXT DEFAULT '{}',
                node_output TEXT DEFAULT '{}',
                iteration INTEGER DEFAULT 0,
                iteration_counters TEXT DEFAULT '{}',
                created_at TEXT
            )"""
        )
        for r in rows:
            db.execute(
                """INSERT INTO checkpoints
                   (id, chain_name, execution_id, node_id, node_index)
                   VALUES (?, ?, ?, ?, ?)""",
                r,
            )


def _seed_legacy_prompts(dir_path, prompts, fragments=None):
    dir_path.mkdir(parents=True, exist_ok=True)
    for slug, versions, aliases in prompts:
        payload = {
            "name": slug,
            "latest_version": versions[-1]["version"],
            "versions": versions,
            "aliases": aliases or {},
        }
        (dir_path / f"{slug}.json").write_text(json.dumps(payload))
    for name, content in (fragments or {}).items():
        (dir_path / f"_fragment_{name}.json").write_text(
            json.dumps({"name": name, "content": content, "version": 1})
        )


class TestMigrator:
    def test_no_op_when_nothing_to_migrate(self, temp_dir):
        report = migrate_to_local_db(
            target_db=temp_dir / "local.db",
            legacy_trace_db=temp_dir / "missing-traces.db",
            legacy_checkpoint_db=temp_dir / "missing-checkpoints.db",
            legacy_prompt_dir=temp_dir / "missing-prompts",
        )
        assert report.nothing_to_do()
        assert report.spans_migrated == 0

    def test_migrates_traces(self, temp_dir):
        legacy = temp_dir / "traces.db"
        _seed_legacy_traces(
            legacy,
            [
                ("s1", "t1", "span-one"),
                ("s2", "t1", "span-two"),
                ("s3", "t2", "span-three"),
            ],
        )
        target = temp_dir / "local.db"
        report = migrate_to_local_db(target_db=target, legacy_trace_db=legacy)
        assert report.spans_migrated == 3
        with SQLiteHelper(target) as db:
            rows = db.fetchall("SELECT COUNT(*) AS n FROM spans")
        assert rows[0]["n"] == 3

    def test_migrates_checkpoints(self, temp_dir):
        legacy = temp_dir / "checkpoints.db"
        _seed_legacy_checkpoints(
            legacy,
            [
                ("cp1", "chain", "exec1", "node1", 0),
                ("cp2", "chain", "exec1", "node2", 1),
            ],
        )
        target = temp_dir / "local.db"
        report = migrate_to_local_db(
            target_db=target, legacy_checkpoint_db=legacy
        )
        assert report.checkpoints_migrated == 2
        with SQLiteHelper(target) as db:
            rows = db.fetchall("SELECT COUNT(*) AS n FROM checkpoints")
        assert rows[0]["n"] == 2

    def test_migrates_prompts_with_versions_and_aliases(self, temp_dir):
        legacy_dir = temp_dir / "legacy_prompts"
        _seed_legacy_prompts(
            legacy_dir,
            prompts=[
                (
                    "greet",
                    [
                        {"name": "greet", "template": "Hi {{name}}", "version": 1,
                         "variables": ["name"], "metadata": {}},
                        {"name": "greet", "template": "Hello {{name}}", "version": 2,
                         "variables": ["name"], "metadata": {}},
                    ],
                    {"production": 1, "staging": 2},
                ),
                (
                    "farewell",
                    [
                        {"name": "farewell", "template": "Bye", "version": 1,
                         "variables": [], "metadata": {}},
                    ],
                    None,
                ),
            ],
            fragments={"tone": "Be friendly."},
        )
        target = temp_dir / "local.db"
        report = migrate_to_local_db(
            target_db=target, legacy_prompt_dir=legacy_dir
        )
        assert report.prompts_migrated == 2
        assert report.prompt_versions_migrated == 3
        assert report.fragments_migrated == 1
        assert report.aliases_migrated == 2

        # Now read them back via the live registry — this is the real proof
        # the migration produced a registry-readable layout.
        from fastaiagent.prompt.registry import PromptRegistry

        reg = PromptRegistry(path=str(target))
        assert reg.load("greet").version == 2
        assert reg.load("greet", version=1).template == "Hi {{name}}"
        assert reg.load("greet", alias="production").version == 1
        assert reg.load("greet", alias="staging").version == 2
        assert reg.load("farewell").template == "Bye"

    def test_is_idempotent(self, temp_dir):
        legacy = temp_dir / "traces.db"
        _seed_legacy_traces(legacy, [("s1", "t1", "one")])
        target = temp_dir / "local.db"
        first = migrate_to_local_db(target_db=target, legacy_trace_db=legacy)
        second = migrate_to_local_db(target_db=target, legacy_trace_db=legacy)
        # Second run still reports rows encountered but does not duplicate
        # them in the target (INSERT OR IGNORE).
        with SQLiteHelper(target) as db:
            rows = db.fetchall("SELECT COUNT(*) AS n FROM spans")
        assert first.spans_migrated == 1
        assert second.spans_migrated == 1  # legacy rows still there
        assert rows[0]["n"] == 1

    def test_skips_source_equal_to_target(self, temp_dir):
        """Guard against accidentally passing the target as a legacy source."""
        target = temp_dir / "local.db"
        # Initialize the target with a span so we can verify it wasn't wiped.
        from fastaiagent.ui.db import init_local_db

        db = init_local_db(target)
        db.execute(
            "INSERT INTO spans (span_id, trace_id, name) VALUES (?, ?, ?)",
            ("pre", "t-pre", "already-here"),
        )
        db.close()

        report = migrate_to_local_db(target_db=target, legacy_trace_db=target)
        assert report.spans_migrated == 0
        assert report.legacy_trace_db is None
