"""Thread-safe SQLite helper for local storage."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _keep_db_perms() -> bool:
    """Resolver for ``FASTAIAGENT_DB_KEEP_PERMS`` (registered in ``ENV_FLAGS``).

    Opting *in* to leaving group/other access on ``local.db`` alone, so an
    unparseable value must mean "not opted in" — the file gets tightened.
    """
    from fastaiagent._internal.env import env_flag

    return env_flag("FASTAIAGENT_DB_KEEP_PERMS", default=False, on_unparsed=False)



def _enable_wal(conn: sqlite3.Connection, db_path: Path) -> None:
    """Put the database in WAL mode, tolerating a concurrent converter.

    ``PRAGMA journal_mode=WAL`` needs exclusive access to convert a rollback
    journal, and it does **not** honour ``busy_timeout`` while doing so — it
    returns SQLITE_BUSY immediately. So several processes opening one fresh
    ``local.db`` at the same instant would race, and all but one died with
    "database is locked" before running a single statement. That is reachable in
    the ordinary local setup, where the UI server and an agent run share a store.

    Journal mode is a property of the FILE, not the connection, so a conversion
    another process is doing is one we do not need to repeat. Read first, retry
    briefly, and if it is still contended, carry on — the winner's WAL applies to
    us too, and a rollback-journal database is slower, not broken.
    """
    for attempt in range(5):
        try:
            row = conn.execute("PRAGMA journal_mode").fetchone()
            if row and str(row[0]).lower() == "wal":
                return
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) and "busy" not in str(exc).lower():
                raise
            time.sleep(0.05 * (attempt + 1))
    logger.debug("Could not switch %s to WAL — another process holds it", db_path)


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
        # ``expanduser`` here and not only in ``SDKConfig.from_env`` because
        # direct callers exist: ``fastaiagent ui --db ~/x/local.db``, the
        # trace/eval CLIs, and user code constructing a helper by hand. Without
        # it a ``~`` path created a literal ``./~/`` directory relative to the
        # current working directory, so the same configured path resolved to a
        # different store depending on where the process was started.
        self.db_path = Path(os.path.expanduser(os.path.expandvars(str(db_path))))
        # ``local.db`` holds trace payloads, prompts, KB contents, and (adjacent)
        # bcrypt hashes, so it must not be group/world readable. SQLite creates
        # files with the process umask (typically 0o644), and pre-v1.49 installs
        # were only tightened at *creation* — so an existing DB stayed
        # world-readable (security_audit_2 N10). We now tighten on every open,
        # but only ever REMOVE group/other access (owner bits are never touched,
        # so the app keeps working), and never when the operator opted out via
        # ``FASTAIAGENT_DB_KEEP_PERMS=1`` (deliberate group sharing).
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._tighten_perms(self.db_path.parent, 0o700)
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
        # Reentrant: ``exclusive()`` holds this for a whole read-modify-write
        # sequence while the statements inside it still go through ``execute()``.
        self._write_lock = threading.RLock()
        # In-flight statement count, so ``close()`` can wait rather than free a
        # connection another thread is executing on. Reads deliberately do NOT
        # take ``_write_lock`` — WAL serves them concurrently with writers — so
        # this counter is the only thing that sees them.
        self._inflight = 0
        self._inflight_cv = threading.Condition()
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

    @staticmethod
    def _tighten_perms(path: Path, mode: int) -> None:
        """Tighten ``path`` to ``mode`` iff it currently grants group/other access.

        Only ever removes access — owner bits are untouched, so the owning
        process keeps working. No-op when the path is already owner-only, and
        when ``FASTAIAGENT_DB_KEEP_PERMS`` is set (operators who deliberately
        share the DB with a group/other). See security_audit_2 N10.
        """
        if _keep_db_perms():
            return
        try:
            current = os.stat(path).st_mode & 0o777
        except OSError:
            return
        if current & 0o077:  # any group/other bit set → tighten
            SQLiteHelper._chmod_quiet(path, mode)

    def _get_conn(self) -> sqlite3.Connection:
        if self._closed:
            raise sqlite3.ProgrammingError("Cannot operate on a closed SQLiteHelper")
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            return conn
        # First time this thread asks for a connection. Note: we keep
        # ``check_same_thread=False`` so :meth:`close` can safely close
        # connections from other threads on shutdown — but every actual
        # query stays in the thread that opened it via TLS.
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # ``busy_timeout`` FIRST, and the order is load-bearing: switching a
        # fresh database to WAL takes a lock of its own, so with the timeout set
        # afterwards that statement had none and returned SQLITE_BUSY the instant
        # another process was converting the same new file. That is the "database
        # is locked" seen when several processes open one ``local.db`` at once.
        conn.execute("PRAGMA busy_timeout=5000")
        _enable_wal(conn, self.db_path)
        # Wait briefly for a competing writer instead of failing immediately with
        # "database is locked". Each SQLiteHelper instance has its own
        # ``_write_lock``, but separate instances (e.g. LocalStorageProcessor vs
        # PlatformSpanExporter's TraceStore) and separate processes (the Local UI
        # server) share only SQLite's file lock — WAL serializes writers, and this
        # timeout absorbs the brief overlap. Strictly safer: it can only turn an
        # immediate error into a short wait.
        # Tighten the DB file on every open (N10) — covers both freshly-created
        # files (SQLite uses the umask, typically 0o644) and pre-existing
        # world-readable DBs from older installs. Only removes group/other bits.
        self._tighten_perms(self.db_path, 0o600)
        with self._connections_lock:
            self._connections.append(conn)
        self._tls.conn = conn
        return conn

    @contextmanager
    def _statement(self) -> Iterator[None]:
        """Count one in-flight statement, so ``close()`` can wait for it.

        Closing a SQLite connection while another thread is executing on it is
        undefined behaviour, and in practice a **segmentation fault** — not an
        exception you can catch. ``check_same_thread=False`` permits the shared
        use; it does not make the close safe.
        """
        with self._inflight_cv:
            if self._closed:
                raise sqlite3.ProgrammingError("Cannot operate on a closed SQLiteHelper")
            self._inflight += 1
        try:
            yield
        finally:
            with self._inflight_cv:
                self._inflight -= 1
                if self._inflight == 0:
                    self._inflight_cv.notify_all()

    def _in_exclusive(self) -> bool:
        return getattr(self._tls, "exclusive_depth", 0) > 0

    @contextmanager
    def exclusive(self) -> Iterator[sqlite3.Connection]:
        """Hold SQLite's write lock for a whole read-modify-write sequence.

        :meth:`execute` commits every statement, which is right for an ordinary
        write and wrong for a schema migration: another **process** can pass its
        own version check in the gap between two of our statements and then
        collide with the schema we just changed. That is not hypothetical — it
        is ``sqlite3.OperationalError: trigger spans_fts_ai already exists``,
        which took down a CI gate and fires in the ordinary local setup where
        the UI and an agent share one ``local.db``.

        ``BEGIN IMMEDIATE`` takes the write lock up front, so the check and the
        change are one step as far as every other process is concerned. Nested
        :meth:`execute` calls inside the block run on the same connection and do
        **not** commit; the block commits once on exit and rolls back on error.
        """
        conn = self._get_conn()
        with self._write_lock:
            depth = getattr(self._tls, "exclusive_depth", 0)
            if depth:  # already inside one — join it rather than nesting BEGINs
                self._tls.exclusive_depth = depth + 1
                try:
                    yield conn
                finally:
                    self._tls.exclusive_depth = depth
                return
            conn.execute("BEGIN IMMEDIATE")
            self._tls.exclusive_depth = 1
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                self._tls.exclusive_depth = 0

    def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Cursor:
        """Execute a SQL statement."""
        conn = self._get_conn()
        with self._statement(), self._write_lock:
            cursor = conn.execute(sql, params)
            if not self._in_exclusive():
                conn.commit()
            return cursor

    def executemany(
        self, sql: str, params_list: list[tuple[Any, ...] | dict[str, Any]]
    ) -> sqlite3.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        conn = self._get_conn()
        with self._statement(), self._write_lock:
            cursor = conn.executemany(sql, params_list)
            if not self._in_exclusive():
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
        with self._statement():
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
        with self._statement():
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def close(self, timeout: float = 5.0) -> None:
        """Close every per-thread connection opened so far.

        **Waits for in-flight statements first.** ``check_same_thread=False``
        lets another thread share these connections — the platform replica's
        drain thread does exactly that — and closing one while it is mid-query
        is undefined behaviour in SQLite. In practice it is a segmentation
        fault, which is not an exception anyone can catch: it takes the whole
        process down, and the crash lands wherever the interpreter happened to
        be rather than here.

        If a statement is still running when ``timeout`` expires, the connection
        is deliberately **left open**. A leaked connection is reclaimed when the
        process exits; a segfault is not recoverable at all.
        """
        with self._inflight_cv:
            self._closed = True
            deadline = time.monotonic() + timeout
            while self._inflight and time.monotonic() < deadline:
                self._inflight_cv.wait(timeout=0.05)
            still_running = self._inflight

        if still_running:
            logger.warning(
                "close() found %d statement(s) still running on %s after %.1fs — "
                "leaving those connections open rather than risking a crash",
                still_running,
                self.db_path,
                timeout,
            )
            return

        with self._connections_lock:
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
