"""Thread-safe SQLite helper for local storage."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SQLiteHelper:
    """Thread-safe SQLite database wrapper.

    Usage:
        db = SQLiteHelper("path/to/db.sqlite")
        db.execute("CREATE TABLE IF NOT EXISTS t (id TEXT, data TEXT)")
        db.execute("INSERT INTO t VALUES (?, ?)", ("id1", "data1"))
        rows = db.fetchall("SELECT * FROM t")
        db.close()

    Or as a context manager:
        with SQLiteHelper("path/to/db.sqlite") as db:
            db.execute("CREATE TABLE ...")
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        # On a fresh install we want the SQLite file (and its containing
        # directory) created with owner-only perms — traces, prompts, KB
        # contents, and bcrypt hashes all live here, and ``local.db`` is
        # otherwise world-readable on POSIX. We tighten *only* on creation
        # so we never silently downgrade perms a user set themselves.
        self._parent_was_new = not self.db_path.parent.exists()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self._parent_was_new:
            self._chmod_quiet(self.db_path.parent, 0o700)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    @staticmethod
    def _chmod_quiet(path: Path, mode: int) -> None:
        """Best-effort ``chmod``. Windows ignores POSIX bits — that's fine."""
        try:
            os.chmod(path, mode)
        except OSError:
            logger.debug("Could not chmod %s to %o", path, mode, exc_info=True)

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            file_was_new = not self.db_path.exists()
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            if file_was_new:
                # SQLite creates the file with the process umask (typically
                # 0o644). Tighten it so trace payloads & bcrypt hashes are
                # not world-readable on shared hosts.
                self._chmod_quiet(self.db_path, 0o600)
        return self._conn

    def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Cursor:
        """Execute a SQL statement."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor

    def executemany(
        self, sql: str, params_list: list[tuple[Any, ...] | dict[str, Any]]
    ) -> sqlite3.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.executemany(sql, params_list)
            conn.commit()
            return cursor

    def fetchone(
        self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()
    ) -> dict[str, Any] | None:
        """Execute a query and return the first row as a dict."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(sql, params)
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    def fetchall(
        self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()
    ) -> list[dict[str, Any]]:
        """Execute a query and return all rows as dicts."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> SQLiteHelper:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
