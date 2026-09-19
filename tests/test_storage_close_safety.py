"""Closing a SQLite connection another thread is using must not crash.

``SQLiteHelper`` opens connections with ``check_same_thread=False`` so ``close()``
can close them from the owning process's main thread. That permits the sharing;
it does **not** make the close safe. Closing a connection while another thread is
mid-query is undefined behaviour in SQLite, and in practice a **segmentation
fault** — which is not an exception anyone can catch, takes the whole process
down, and surfaces wherever the interpreter happened to be rather than at the
close.

This is not hypothetical. It took down CI twice, on two branches and two
operating systems, both times immediately after ``tests/test_restore_if_missing``
— the suite that exercises the platform replica, whose drain thread shares a
helper with the test's main thread. It reads as "exit code 139" and names
whatever test the runner was on.

The crash cannot be asserted in-process, because it kills the process doing the
asserting. So the risky scenario runs in a **subprocess** and the test asserts
the exit code — ``-11`` is SIGSEGV.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# A worker thread hammering reads while the main thread closes underneath it.
_CRASHER = textwrap.dedent(
    """
    import os, sys, threading, time
    sys.path.insert(0, {repo!r})
    from fastaiagent._internal.storage import SQLiteHelper

    db = os.path.join({tmp!r}, "race.db")
    h = SQLiteHelper(db)
    h.execute("CREATE TABLE t (a INTEGER)")
    for i in range(500):
        h.execute("INSERT INTO t VALUES (?)", (i,))

    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                h.fetchall("SELECT a FROM t")
            except Exception:
                # A closed helper raising is the CORRECT outcome. Only a crash
                # is the failure, and a crash never reaches this handler.
                return

    threads = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.25)          # let them get properly in flight
    h.close()                 # <-- frees connections those threads are using
    stop.set()
    for t in threads:
        t.join(timeout=5)
    print("survived")
    """
)


@pytest.mark.parametrize("run", range(3))
def test_closing_under_concurrent_readers_does_not_crash(tmp_path, run: int) -> None:
    """A guard, and honestly labelled: this has never gone red on a dev machine.

    Five runs with 16 readers over a 5000-row table did not crash on macOS with
    this SQLite build, before the fix. The crash is real — it took down CI twice
    at the same point, on two operating systems — but it needs a timing this
    machine does not reproduce. What justifies the fix is the code, not this
    test: freeing a connection another thread is executing on is undefined
    behaviour whether or not it happens to survive.

    The test earns its place as a regression guard on the platforms that DO hit
    it, and repeats because a race proves more over three runs than one.
    ``test_close_waits_for_an_in_flight_statement`` below is the red-before,
    green-after proof of the mechanism.
    """
    repo = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    script = _CRASHER.format(repo=repo, tmp=str(tmp_path))

    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )

    assert proc.returncode == 0, (
        f"exit {proc.returncode}"
        + (
            " (SIGSEGV — a connection was freed mid-query)"
            if proc.returncode == -11
            else " (see stderr)"
        )
        + f"\nstdout: {proc.stdout}\nstderr: {proc.stderr[-2000:]}"
    )
    assert "survived" in proc.stdout


def test_close_waits_for_an_in_flight_statement(tmp_path) -> None:
    """The mechanism, asserted directly rather than inferred from not crashing."""
    import threading
    import time

    from fastaiagent._internal.storage import SQLiteHelper

    db = SQLiteHelper(str(tmp_path / "wait.db"))
    db.execute("CREATE TABLE t (a INTEGER)")

    entered = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def slow_statement() -> None:
        # Hold the in-flight count open across a wait, standing in for a long
        # query on a big table.
        with db._statement():
            entered.set()
            release.wait(timeout=10)
            order.append("statement-done")

    worker = threading.Thread(target=slow_statement, daemon=True)
    worker.start()
    assert entered.wait(timeout=5)

    def do_close() -> None:
        db.close(timeout=10)
        order.append("close-returned")

    closer = threading.Thread(target=do_close, daemon=True)
    closer.start()
    time.sleep(0.2)
    assert order == [], "close() returned while a statement was still in flight"

    release.set()
    worker.join(timeout=5)
    closer.join(timeout=5)
    assert order == ["statement-done", "close-returned"]


def test_close_gives_up_rather_than_freeing_a_live_connection(tmp_path) -> None:
    """A statement that never finishes must not turn into a crash.

    The connection is deliberately left open: a leaked connection is reclaimed
    when the process exits, a segfault is not recoverable at all.
    """
    import threading

    from fastaiagent._internal.storage import SQLiteHelper

    db = SQLiteHelper(str(tmp_path / "stuck.db"))
    db.execute("CREATE TABLE t (a INTEGER)")

    entered = threading.Event()
    release = threading.Event()

    def stuck() -> None:
        with db._statement():
            entered.set()
            release.wait(timeout=30)

    worker = threading.Thread(target=stuck, daemon=True)
    worker.start()
    assert entered.wait(timeout=5)

    db.close(timeout=0.2)  # must return, not hang and not crash
    assert db._closed is True

    release.set()
    worker.join(timeout=5)


def test_a_closed_helper_refuses_new_statements(tmp_path) -> None:
    """Closing still means closed — the fix must not turn it into a no-op."""
    import sqlite3

    from fastaiagent._internal.storage import SQLiteHelper

    db = SQLiteHelper(str(tmp_path / "closed.db"))
    db.execute("CREATE TABLE t (a INTEGER)")
    db.close()

    with pytest.raises(sqlite3.ProgrammingError):
        db.fetchall("SELECT a FROM t")
