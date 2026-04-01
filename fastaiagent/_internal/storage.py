"""Thread-safe SQLite helper for local storage."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any


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
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        """Execute a SQL statement."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor

    def executemany(self, sql: str, params_list: list[tuple | dict]) -> sqlite3.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.executemany(sql, params_list)
            conn.commit()
            return cursor

    def fetchone(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        """Execute a query and return the first row as a dict."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(sql, params)
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    def fetchall(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
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
