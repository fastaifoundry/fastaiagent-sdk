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
        # security_review_1.md M7 — connections are now per-thread.
        # The previous design used a single shared connection guarded by a
        # ``threading.Lock`` and ``check_same_thread=False``. That worked
        # for the existing routes but was fragile: any future code path
        # that bypassed the lock (or even released it across an ``await``)
        # could corrupt ``local.db``. With per-thread connections each
        # thread owns its own ``sqlite3.Connection``; SQLite's WAL mode
        # already serializes writers correctly, and a small write lock
        # keeps SQLITE_BUSY churn down under contention.
        self._tls = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.Lock()
        self._write_lock = threading.Lock()
        # Backwards-compat alias: SQLiteCheckpointer (and possibly user
        # code) reaches into ``db._lock`` to wrap a multi-statement
        # transaction across two ``conn.execute`` calls. Pre-M7 the
        # helper exposed a single ``_lock``; we keep that name pointing
        # at the same lock object so existing callers keep working.
        self._lock = self._write_lock
        self._closed = False

    @staticmethod
    def _chmod_quiet(path: Path, mode: int) -> None:
        """Best-effort ``chmod``. Windows ignores POSIX bits — that's fine."""
        try:
            os.chmod(path, mode)
        except OSError:
            logger.debug("Could not chmod %s to %o", path, mode, exc_info=True)

    def _get_conn(self) -> sqlite3.Connection:
        if self._closed:
            raise sqlite3.ProgrammingError(
                "Cannot operate on a closed SQLiteHelper"
            )
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            return conn
        # First time this thread asks for a connection. Note: we keep
        # ``check_same_thread=False`` so :meth:`close` can safely close
        # connections from other threads on shutdown — but every actual
        # query stays in the thread that opened it via TLS.
        file_was_new = not self.db_path.exists()
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        if file_was_new:
            # SQLite creates the file with the process umask (typically
            # 0o644). Tighten it so trace payloads & bcrypt hashes are
            # not world-readable on shared hosts.
            self._chmod_quiet(self.db_path, 0o600)
        with self._connections_lock:
            self._connections.append(conn)
        self._tls.conn = conn
        return conn

    def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Cursor:
        """Execute a SQL statement."""
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor

    def executemany(
        self, sql: str, params_list: list[tuple[Any, ...] | dict[str, Any]]
    ) -> sqlite3.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.executemany(sql, params_list)
            conn.commit()
            return cursor

    def fetchone(
        self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()
    ) -> dict[str, Any] | None:
        """Execute a query and return the first row as a dict."""
        # Reads run on the calling thread's own connection — SQLite WAL
        # serves them concurrently with writers, so we don't take the
        # write lock here.
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
        conn = self._get_conn()
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close every per-thread connection opened so far."""
        with self._connections_lock:
            self._closed = True
            for conn in self._connections:
                try:
                    conn.close()
                except sqlite3.Error:
                    logger.debug("close() failed on a per-thread connection", exc_info=True)
            self._connections.clear()
        # Drop the TLS holder so any later reuse fails fast (rather than
        # quietly opening a fresh connection on a closed helper).
        self._tls = threading.local()

    def __enter__(self) -> SQLiteHelper:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
