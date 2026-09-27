"""Pluggable fact-store backends behind a single ``FactStore`` protocol.

``MemoryStore`` (SQLite) is the default and structurally satisfies the protocol.
:class:`PostgresFactStore` and :class:`RedisFactStore` are drop-in external
backends for multi-node / multi-user deployments, selected via
``Memory(location="postgres://…" | "redis://…")``.

All backends share the same **safe-by-default scoping** contract as
:meth:`MemoryStore.list_active` / :meth:`MemoryStore.delete`:

- at ``user`` / ``project`` scope, an empty ``scope_id`` reads/refuses (never
  every subject); ``scope_id="*"`` opts into all;
- ``agent`` scope is permissive (the global tier).

Facts are idempotent on ``(scope, scope_id, fact, project_id)`` and versioned by
``supersede`` (not overwrite), exactly like the SQLite store.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
import uuid
import weakref
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from fastaiagent.learn.store import Fact, Scope

_log = logging.getLogger(__name__)


@runtime_checkable
class FactStore(Protocol):
    """The interface every fact backend implements (``MemoryStore`` conforms)."""

    def add(self, fact: Fact) -> int: ...
    def get(self, fact_id: int) -> Fact | None: ...
    def list_active(
        self, scope: Scope, scope_id: str = "", project_id: str = "", limit: int | None = None
    ) -> list[Fact]: ...
    def supersede(self, old_id: int, new_id: int) -> None: ...
    def delete(
        self, scope: Scope, scope_id: str = "", project_id: str = "", fact: str | None = None
    ) -> int: ...


def make_fact_store(location: str):
    """Resolve a ``location`` string to a backend instance."""
    if location.startswith(("postgres://", "postgresql://")):
        return PostgresFactStore(location)
    if location.startswith(("redis://", "rediss://")):
        return RedisFactStore(location)
    raise ValueError(f"unsupported fact-store location: {location!r}")


# ---------------------------------------------------------------------------
# Semantic layer — vector-index facts for meaning-based retrieve(query)
# ---------------------------------------------------------------------------


_FACT_VECTOR_NS = uuid.UUID("5b0f3a52-6c1e-4f5e-9a3d-2f4b7c1d8e90")


def _vector_id(fact_id: int, project_id: str) -> str:
    """A stable UUID for a fact's vector — valid on every vector backend.

    Qdrant accepts only unsigned integers or UUIDs as point ids, so the old
    ``str(fact_id)`` (``"1"``) was rejected there and nothing was indexed.
    """
    return str(uuid.uuid5(_FACT_VECTOR_NS, f"{project_id}:{fact_id}"))


def _chunk_fact_id(chunk: Any) -> int | None:
    """The fact a vector belongs to: ``metadata["fact_id"]``, or the numeric
    chunk id vectors written before 1.81.0 used."""
    raw = (chunk.metadata or {}).get("fact_id", chunk.id)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


class SemanticFactStore:
    """Wrap any :class:`FactStore` and mirror every fact into a ``VectorStore``.

    Delegates the full ``FactStore`` contract to ``inner`` and, on ``add``, also
    embeds the fact text and indexes it by id — so facts written *either* via
    ``Memory.persist`` *or* by ``FactExtractionBlock`` (which shares this store
    handle) become semantically searchable. :meth:`search` returns
    ``(Fact, score)`` for a query within a scope, honoring safe-by-default
    scoping and skipping superseded rows.

    The store is the source of truth, not the index: before each search the
    subject's active facts are read from ``inner`` and any this process hasn't
    indexed are embedded — so a restarted process (an in-process FAISS index
    starts empty) and facts written by another process are both found.
    """

    def __init__(self, inner, index, embedder):
        self._inner = inner
        self._index = index
        self._embedder = embedder
        self._indexed: set[int] = set()

    def _index_facts(self, facts: list[Fact]) -> None:
        """Embed and index facts this process hasn't indexed yet (one batch)."""
        from fastaiagent.kb.chunking import Chunk

        todo = [f for f in facts if f.id is not None and f.id not in self._indexed]
        if not todo:
            return
        try:
            embeddings = self._embedder.embed([f.fact for f in todo])
            chunks = [
                Chunk(
                    id=_vector_id(f.id, f.project_id),  # type: ignore[arg-type]
                    content=f.fact,
                    metadata={
                        "fact_id": f.id,
                        "scope": f.scope,
                        "scope_id": f.scope_id,
                        "project_id": f.project_id,
                    },
                    index=0,
                    start_char=0,
                    end_char=len(f.fact),
                )
                for f in todo
            ]
            self._index.add(chunks, embeddings)
            self._indexed.update(f.id for f in todo if f.id is not None)
        except Exception:
            _log.warning(
                "SemanticFactStore: failed to index facts %s",
                [f.id for f in todo],
                exc_info=True,
            )

    # -- FactStore delegation (+ indexing on add) --
    def add(self, fact: Fact) -> int:
        fid = self._inner.add(fact)
        if fid not in self._indexed:
            stored = self._inner.get(fid) or fact
            if stored.id is None:
                stored = dataclasses.replace(stored, id=fid)
            self._index_facts([stored])
        return fid

    def get(self, fact_id: int) -> Fact | None:
        return self._inner.get(fact_id)

    def list_active(self, scope, scope_id="", project_id="", limit=None):
        return self._inner.list_active(scope, scope_id, project_id, limit)

    def supersede(self, old_id: int, new_id: int) -> None:
        self._inner.supersede(old_id, new_id)

    def delete(self, scope, scope_id="", project_id="", fact=None) -> int:
        # Drop matching vectors before the rows disappear. A failure here leaves
        # stale vectors that ``search`` skips (their fact is gone), so it is
        # logged rather than raised.
        try:
            doomed = [
                f
                for f in self._inner.list_active(scope, scope_id, project_id)
                if (fact is None or f.fact == fact) and f.id is not None
            ]
            if doomed:
                self._index.delete([_vector_id(f.id, f.project_id) for f in doomed])
                self._indexed.difference_update(f.id for f in doomed)
        except Exception:
            _log.warning("SemanticFactStore: failed to remove vectors", exc_info=True)
        return self._inner.delete(scope, scope_id, project_id, fact)

    # -- Semantic search --
    def search(
        self, query: str, scope: Scope, scope_id: str = "", project_id: str = "", top_k: int = 10
    ) -> list[tuple[Fact, float]]:
        """Return ``(Fact, score)`` for facts semantically matching ``query``."""
        if scope in ("user", "project") and scope_id == "":
            return []
        self._index_facts(self._inner.list_active(scope, scope_id, project_id))
        emb = self._embedder.embed([query])[0]

        def own(chunk: Any) -> bool:
            m = chunk.metadata or {}
            if m.get("scope") != scope or m.get("project_id", "") != project_id:
                return False
            return not (scope_id and scope_id != "*" and m.get("scope_id") != scope_id)

        # Other subjects share the index; widen the search while they crowd out
        # this one's facts, up to the index's size.
        total = _index_count(self._index)
        ceiling = total if total is not None else top_k * 256
        fetch = min(top_k * 5, max(ceiling, top_k))
        while True:
            hits = list(self._index.search(emb, fetch))
            out: list[tuple[Fact, float]] = []
            seen: set[int] = set()
            for chunk, score in hits:
                fid = _chunk_fact_id(chunk)
                if fid is None or fid in seen or not own(chunk):
                    continue
                f = self._inner.get(fid)
                if f is None or f.superseded_by is not None:
                    continue
                seen.add(fid)
                out.append((f, float(score)))
                if len(out) >= top_k:
                    return out
            if len(hits) < fetch or fetch >= ceiling:
                return out
            fetch = min(fetch * 2, ceiling)


def _index_count(index: Any) -> int | None:
    fn = getattr(index, "count", None)
    if not callable(fn):
        return None
    try:
        return int(fn())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------

_PG_DDL = """
CREATE TABLE IF NOT EXISTS learned_memory (
    id              BIGSERIAL PRIMARY KEY,
    scope           TEXT NOT NULL,
    scope_id        TEXT NOT NULL DEFAULT '',
    fact            TEXT NOT NULL,
    source_trace_id TEXT,
    confidence      DOUBLE PRECISION DEFAULT 1.0,
    created_at      DOUBLE PRECISION NOT NULL,
    superseded_by   BIGINT,
    project_id      TEXT NOT NULL DEFAULT '',
    UNIQUE (scope, scope_id, fact, project_id)
);
-- 1.82.0: "learned" for a fact the SDK extracted, "" for one written directly.
ALTER TABLE learned_memory ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT '';
-- The per-turn read is "the newest N active facts" for one subject, or for a
-- whole scope; these let it stop after N rows instead of sorting them all.
CREATE INDEX IF NOT EXISTS idx_learned_memory_subject_active
    ON learned_memory (scope, scope_id, project_id, created_at DESC)
    WHERE superseded_by IS NULL;
CREATE INDEX IF NOT EXISTS idx_learned_memory_scope_active
    ON learned_memory (scope, project_id, created_at DESC)
    WHERE superseded_by IS NULL;
"""


def _close_pool_in(pool: Any, pid: int) -> None:
    """Close ``pool`` if this is the process that opened it (a finalizer)."""
    if os.getpid() == pid:
        pool.close()


class PostgresFactStore:
    """``FactStore`` over Postgres (via ``psycopg`` v3). Requires the
    ``fastaiagent[postgres]`` extra. Table is created on first use.

    Connections come from a pool of ``min_pool_size`` to ``max_pool_size``,
    one pool per process: a store built before a fork opens its own pool in
    the child. ``close()`` releases the connections; the store reopens on its
    next use.
    """

    def __init__(self, dsn: str, *, min_pool_size: int = 1, max_pool_size: int = 10):
        try:
            import psycopg
            from psycopg_pool import ConnectionPool  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "PostgresFactStore needs psycopg — install fastaiagent[postgres]"
            ) from e
        self._dsn = dsn
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: Any = None
        self._pool_pid = 0
        self._pool_finalizer: weakref.finalize[Any, Any] | None = None
        self._pool_lock = threading.Lock()
        # A direct connection, not the pool: a pool connects in the background,
        # so a bad DSN would only surface as a pool timeout on first use.
        with psycopg.connect(dsn) as c:
            c.execute(_PG_DDL)
            c.commit()

    def _get_pool(self) -> Any:
        pid = os.getpid()
        pool = self._pool
        if pool is not None and self._pool_pid == pid:
            return pool
        with self._pool_lock:
            if self._pool is None or self._pool_pid != pid:
                from psycopg_pool import ConnectionPool

                # After a fork the parent's pool is left alone — never used or
                # closed here: its sockets belong to the parent.
                pool = ConnectionPool(
                    conninfo=self._dsn,
                    min_size=self._min_pool_size,
                    max_size=self._max_pool_size,
                    open=False,
                    # A pooled connection can outlive a server restart; check
                    # it on the way out, as a fresh connection per call did.
                    check=getattr(ConnectionPool, "check_connection", None),
                )
                pool.open()
                self._pool, self._pool_pid = pool, pid
                # Close the pool when this store is collected (and at exit), in
                # the thread that drops it. Left to psycopg_pool's own __del__,
                # the close can land on one of the pool's worker threads and
                # fail with "cannot join current thread".
                self._pool_finalizer = weakref.finalize(self, _close_pool_in, pool, pid)
            return self._pool

    def _conn(self) -> Any:
        return self._get_pool().connection()

    def close(self) -> None:
        """Close this process's connections. The store reopens on next use."""
        with self._pool_lock:
            pool, self._pool = self._pool, None
            finalizer, self._pool_finalizer = self._pool_finalizer, None
            ours = self._pool_pid == os.getpid()
        if finalizer is not None:
            finalizer.detach()
        if pool is not None and ours:
            pool.close()

    @staticmethod
    def _row_to_fact(r: tuple) -> Fact:
        return Fact(
            id=r[0],
            scope=r[1],
            scope_id=r[2],
            fact=r[3],
            source_trace_id=r[4],
            confidence=r[5],
            created_at=r[6],
            superseded_by=r[7],
            project_id=r[8],
            source=r[9] or "",
        )

    _COLS = (
        "id, scope, scope_id, fact, source_trace_id, "
        "confidence, created_at, superseded_by, project_id, source"
    )

    def add(self, fact: Fact) -> int:
        if not fact.fact.strip():
            raise ValueError("fact text must be non-empty")
        if fact.scope not in ("user", "project", "agent"):
            raise ValueError(f"scope must be one of user|project|agent, got {fact.scope!r}")
        created = fact.created_at if fact.created_at is not None else time.time()
        with self._conn() as c:
            cur = c.execute(
                "SELECT id FROM learned_memory "
                "WHERE scope=%s AND scope_id=%s AND fact=%s AND project_id=%s",
                (fact.scope, fact.scope_id, fact.fact, fact.project_id),
            )
            row = cur.fetchone()
            if row:
                return int(row[0])
            # ON CONFLICT: a writer that inserted the same fact since the SELECT
            # wins, and this one returns its row instead of raising
            # UniqueViolation. No conflict target, so a table without the
            # constraint still accepts the insert.
            row = c.execute(
                "INSERT INTO learned_memory "
                "(scope, scope_id, fact, source_trace_id, confidence, created_at, project_id, "
                "source) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING id",
                (
                    fact.scope,
                    fact.scope_id,
                    fact.fact,
                    fact.source_trace_id,
                    fact.confidence,
                    created,
                    fact.project_id,
                    fact.source,
                ),
            ).fetchone()
            if row is None:
                row = c.execute(
                    "SELECT id FROM learned_memory "
                    "WHERE scope=%s AND scope_id=%s AND fact=%s AND project_id=%s",
                    (fact.scope, fact.scope_id, fact.fact, fact.project_id),
                ).fetchone()
            c.commit()
            return int(row[0])

    def get(self, fact_id: int) -> Fact | None:
        with self._conn() as c:
            cur = c.execute(f"SELECT {self._COLS} FROM learned_memory WHERE id=%s", (fact_id,))
            row = cur.fetchone()
            return self._row_to_fact(row) if row else None

    def list_active(
        self, scope: Scope, scope_id: str = "", project_id: str = "", limit: int | None = None
    ) -> list[Fact]:
        if scope in ("user", "project") and scope_id == "":
            return []
        sql = (
            f"SELECT {self._COLS} FROM learned_memory "
            "WHERE scope=%s AND project_id=%s AND superseded_by IS NULL"
        )
        params: list = [scope, project_id]
        if scope_id and scope_id != "*":
            sql += " AND scope_id=%s"
            params.append(scope_id)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with self._conn() as c:
            return [self._row_to_fact(r) for r in c.execute(sql, tuple(params)).fetchall()]

    def supersede(self, old_id: int, new_id: int) -> None:
        with self._conn() as c:
            got = c.execute(
                "SELECT count(*) FROM learned_memory WHERE id IN (%s,%s)", (old_id, new_id)
            ).fetchone()[0]
            if got != 2:
                raise ValueError(f"supersede: missing row(s) old_id={old_id} new_id={new_id}")
            c.execute("UPDATE learned_memory SET superseded_by=%s WHERE id=%s", (new_id, old_id))
            c.commit()

    def delete(
        self, scope: Scope, scope_id: str = "", project_id: str = "", fact: str | None = None
    ) -> int:
        if scope in ("user", "project") and scope_id == "":
            raise ValueError('delete at user/project scope needs an explicit scope_id (or "*")')
        sql = "DELETE FROM learned_memory WHERE scope=%s AND project_id=%s"
        params: list = [scope, project_id]
        if scope_id and scope_id != "*":
            sql += " AND scope_id=%s"
            params.append(scope_id)
        if fact is not None:
            sql += " AND fact=%s"
            params.append(fact)
        with self._conn() as c:
            cur = c.execute(sql, tuple(params))
            n = cur.rowcount or 0
            c.commit()
            return int(n)


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


# Bumped when the Redis layout gains an index that older data lacks. A store
# that opens a namespace without this marker indexes it once (``reindex``).
_REDIS_SCHEMA = "2"


class RedisFactStore:
    """``FactStore`` over Redis. Requires the ``redis`` package.

    Layout: each fact is a hash ``fa:fact:{id}``; ids are minted from
    ``fa:fact:seq``; ``fa:uniq`` maps the idempotency tuple → id; active ids are
    tracked in per-``(scope,scope_id,project)`` sets (``fa:act:…``) with the set
    of scope_ids per ``(scope,project)`` in ``fa:sids:…`` so deletes can fan out.

    Reads go through two newest-first sorted indexes of active ids, scored by
    ``created_at``: per subject (``fa:zact:…``) and per ``(scope,project)``
    (``fa:zall:…``) for ``scope_id="*"`` and permissive ``agent`` reads. A read
    fetches only the ``limit`` facts it returns, however many are stored.
    """

    def __init__(self, url: str, *, namespace: str = "fa"):
        try:
            import redis
        except ImportError as e:  # pragma: no cover
            raise ImportError("RedisFactStore needs the redis package: pip install redis") from e
        self._r = redis.from_url(url, decode_responses=True)
        self._ns = namespace
        if self._r.get(self._k("schema")) != _REDIS_SCHEMA:
            self.reindex()

    def _k(self, *parts: str) -> str:
        return ":".join((self._ns, *parts))

    def _act_key(self, scope: str, scope_id: str, project_id: str) -> str:
        return self._k("act", scope, scope_id, project_id)

    def _all_key(self, scope: str, scope_id: str, project_id: str) -> str:
        return self._k("all", scope, scope_id, project_id)

    def _sids_key(self, scope: str, project_id: str) -> str:
        return self._k("sids", scope, project_id)

    def _zsubject_key(self, scope: str, scope_id: str, project_id: str) -> str:
        return self._k("zact", scope, scope_id, project_id)

    def _zscope_key(self, scope: str, project_id: str) -> str:
        return self._k("zall", scope, project_id)

    def _index(self, fid: int, scope: str, scope_id: str, project_id: str, created: float) -> None:
        self._r.zadd(self._zsubject_key(scope, scope_id, project_id), {str(fid): created})
        self._r.zadd(self._zscope_key(scope, project_id), {str(fid): created})

    def _unindex(self, fid: int, scope: str, scope_id: str, project_id: str) -> None:
        self._r.zrem(self._zsubject_key(scope, scope_id, project_id), str(fid))
        self._r.zrem(self._zscope_key(scope, project_id), str(fid))

    def _get_many(self, ids: Iterable[Any]) -> list[Fact]:
        """Fetch fact hashes in one round trip, keeping ``ids`` order."""
        pipe = self._r.pipeline(transaction=False)
        for fid in ids:
            pipe.hgetall(self._k("fact", str(fid)))
        return [self._hash_to_fact(d) for d in pipe.execute() if d]

    def reindex(self) -> int:
        """Build the sorted read indexes from the active-id sets.

        Runs once, automatically, when this version first opens a namespace an
        older SDK wrote. Re-run it if an older SDK kept writing to the namespace
        after that — its new facts are missing from the indexes until you do.
        Only adds entries, so it is safe to repeat. Returns the facts indexed.
        """
        indexed = 0
        for key in self._r.scan_iter(match=self._k("act", "*")):
            for f in self._get_many(self._r.smembers(key)):
                if f.superseded_by is None and f.id is not None:
                    self._index(f.id, f.scope, f.scope_id, f.project_id, f.created_at or 0.0)
                    indexed += 1
        self._r.set(self._k("schema"), _REDIS_SCHEMA)
        return indexed

    @staticmethod
    def _uniq(scope: str, scope_id: str, fact: str, project_id: str) -> str:
        import hashlib

        h = hashlib.sha256(f"{scope}\x00{scope_id}\x00{fact}\x00{project_id}".encode()).hexdigest()
        return h

    def _hash_to_fact(self, d: dict) -> Fact:
        return Fact(
            id=int(d["id"]),
            scope=d["scope"],
            scope_id=d["scope_id"],
            fact=d["fact"],
            source_trace_id=d.get("source_trace_id") or None,
            confidence=float(d.get("confidence", 1.0)),
            created_at=float(d["created_at"]),
            superseded_by=int(d["superseded_by"]) if d.get("superseded_by") else None,
            project_id=d.get("project_id", ""),
            source=d.get("source", ""),
        )

    def add(self, fact: Fact) -> int:
        if not fact.fact.strip():
            raise ValueError("fact text must be non-empty")
        if fact.scope not in ("user", "project", "agent"):
            raise ValueError(f"scope must be one of user|project|agent, got {fact.scope!r}")
        uniq = self._k("uniq", self._uniq(fact.scope, fact.scope_id, fact.fact, fact.project_id))
        existing = self._r.get(uniq)
        if existing:
            return int(existing)
        fid = int(self._r.incr(self._k("fact", "seq")))
        created = fact.created_at if fact.created_at is not None else time.time()
        fact_key = self._k("fact", str(fid))
        self._r.hset(
            fact_key,
            mapping={
                "id": fid,
                "scope": fact.scope,
                "scope_id": fact.scope_id,
                "fact": fact.fact,
                "source_trace_id": fact.source_trace_id or "",
                "confidence": fact.confidence,
                "created_at": created,
                "superseded_by": "",
                "project_id": fact.project_id,
                "source": fact.source,
            },
        )
        # Claim the fact only if no concurrent writer has: SET NX is atomic, so
        # exactly one writer's row is indexed and the others return its id.
        if not self._r.set(uniq, fid, nx=True):
            winner = self._r.get(uniq)
            if winner is not None:
                self._r.delete(fact_key)  # never indexed, so never read
                return int(winner)
            # The winner was deleted in between: this writer's row stands.
            self._r.set(uniq, fid)
        self._r.sadd(self._act_key(fact.scope, fact.scope_id, fact.project_id), fid)
        self._r.sadd(self._all_key(fact.scope, fact.scope_id, fact.project_id), fid)
        self._r.sadd(self._sids_key(fact.scope, fact.project_id), fact.scope_id)
        self._index(fid, fact.scope, fact.scope_id, fact.project_id, created)
        return fid

    def get(self, fact_id: int) -> Fact | None:
        d = self._r.hgetall(self._k("fact", str(fact_id)))
        return self._hash_to_fact(d) if d else None

    def _ids(self, kind: str, scope: str, scope_id: str, project_id: str) -> set[str]:
        """Collect ids from the ``act`` (active) or ``all`` set(s) for a scope."""
        key = self._act_key if kind == "act" else self._all_key
        if scope_id and scope_id != "*":
            return set(self._r.smembers(key(scope, scope_id, project_id)))
        # "*" or permissive agent-empty: fan out over every scope_id in scope+project
        ids: set[str] = set()
        for sid in self._r.smembers(self._sids_key(scope, project_id)):
            ids |= set(self._r.smembers(key(scope, sid, project_id)))
        return ids

    def list_active(
        self, scope: Scope, scope_id: str = "", project_id: str = "", limit: int | None = None
    ) -> list[Fact]:
        if scope in ("user", "project") and scope_id == "":
            return []
        if limit is not None and limit <= 0:
            return []
        if scope_id and scope_id != "*":
            key = self._zsubject_key(scope, scope_id, project_id)
        else:  # "*", or the permissive agent-wide read
            key = self._zscope_key(scope, project_id)
        ids = self._r.zrevrange(key, 0, -1 if limit is None else limit - 1)
        # An older SDK still writing here does not unindex what it supersedes.
        return [f for f in self._get_many(ids) if f.superseded_by is None]

    def supersede(self, old_id: int, new_id: int) -> None:
        old = self.get(old_id)
        new = self.get(new_id)
        if not old or not new:
            raise ValueError(f"supersede: missing row(s) old_id={old_id} new_id={new_id}")
        self._r.hset(self._k("fact", str(old_id)), "superseded_by", new_id)
        self._r.srem(self._act_key(old.scope, old.scope_id, old.project_id), old_id)
        self._unindex(old_id, old.scope, old.scope_id, old.project_id)

    def delete(
        self, scope: Scope, scope_id: str = "", project_id: str = "", fact: str | None = None
    ) -> int:
        if scope in ("user", "project") and scope_id == "":
            raise ValueError('delete at user/project scope needs an explicit scope_id (or "*")')
        n = 0
        # Delete ALL matching rows (incl. superseded history) — a true forget.
        for fid in list(self._ids("all", scope, scope_id, project_id)):
            f = self.get(int(fid))
            if f is None or (fact is not None and f.fact != fact):
                continue
            self._r.delete(self._k("fact", str(fid)))
            self._r.srem(self._act_key(f.scope, f.scope_id, f.project_id), fid)
            self._r.srem(self._all_key(f.scope, f.scope_id, f.project_id), fid)
            self._unindex(int(fid), f.scope, f.scope_id, f.project_id)
            self._r.delete(self._k("uniq", self._uniq(f.scope, f.scope_id, f.fact, f.project_id)))
            n += 1
        return n
