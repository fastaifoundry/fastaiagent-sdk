"""Point the backend-gated integration tests at local dev containers.

``test_postgres_checkpointer.py``, ``test_postgres_concurrent_resume.py`` and
``test_faststore_conformance.py`` each read ``PG_TEST_DSN`` / ``REDIS_TEST_URL``
at import time and skip the non-SQLite half of their matrix when unset. On a
developer machine those variables are essentially never exported, so 39 of the
suite's 50 skips were these three files — including the headline behaviour of
1.65.0, that re-using a ``checkpoint_id`` raises on ``PostgresCheckpointer``.
Per CLAUDE.md §3, *a skip is not a check*.

This conftest closes that by probing the DSNs that those modules' own
docstrings tell you to start (``scripts/dev_backends.sh up``) and exporting
them when — and only when — something is actually listening.

Two deliberate constraints:

* **An explicit ``PG_TEST_DSN`` / ``REDIS_TEST_URL`` always wins.** CI sets
  them; this never overrides.
* **The Postgres database must be named ``fastaiagent_test``.** An
  unconditional fallback to port 5432 would point the suite at whatever
  Postgres a developer happens to be running, and these tests create and drop
  tables. The name check keeps the blast radius on a container that exists for
  this purpose.

Failing to connect is not an error: the variable stays unset and the tests skip
exactly as they did before.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from urllib.parse import urlsplit

# The DSNs printed by ``scripts/dev_backends.sh up`` and named in each gated
# module's docstring. Non-default ports so they cannot collide with a local
# Postgres or Redis a developer runs for their own work.
_DEFAULT_PG_DSN = "postgresql://postgres:test@127.0.0.1:55432/fastaiagent_test"
_DEFAULT_REDIS_URL = "redis://127.0.0.1:56379/15"

# Only a database dedicated to this suite may be auto-detected; these tests
# create, populate and drop tables.
_REQUIRED_PG_DBNAME = "fastaiagent_test"

_PROBE_TIMEOUT_SECONDS = 2


def _postgres_reachable(dsn: str) -> bool:
    """True when ``dsn`` accepts a connection and names the test database."""
    if urlsplit(dsn).path.lstrip("/") != _REQUIRED_PG_DBNAME:
        return False
    try:
        import psycopg
    except ImportError:
        return False
    try:
        with psycopg.connect(dsn, connect_timeout=_PROBE_TIMEOUT_SECONDS) as conn:
            conn.execute("SELECT 1")
    except Exception:
        return False
    return True


def _redis_reachable(url: str) -> bool:
    """True when ``url`` answers a PING."""
    try:
        import redis
    except ImportError:
        return False
    try:
        client = redis.Redis.from_url(
            url,
            socket_connect_timeout=_PROBE_TIMEOUT_SECONDS,
            socket_timeout=_PROBE_TIMEOUT_SECONDS,
        )
        client.ping()
        client.close()
    except Exception:
        return False
    return True


def _autodetect(env_var: str, default: str, reachable: Callable[[str], bool]) -> None:
    if os.environ.get(env_var):
        return  # an explicit setting always wins
    if reachable(default):
        os.environ[env_var] = default


_autodetect("PG_TEST_DSN", _DEFAULT_PG_DSN, _postgres_reachable)
_autodetect("REDIS_TEST_URL", _DEFAULT_REDIS_URL, _redis_reachable)
