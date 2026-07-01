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

import time
from typing import Protocol, runtime_checkable

from fastaiagent.learn.store import Fact, Scope


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


class SemanticFactStore:
    """Wrap any :class:`FactStore` and mirror every fact into a ``VectorStore``.

    Delegates the full ``FactStore`` contract to ``inner`` and, on ``add``, also
    embeds the fact text and indexes it by id — so facts written *either* via
    ``Memory.persist`` *or* by ``FactExtractionBlock`` (which shares this store
    handle) become semantically searchable. :meth:`search` returns
    ``(Fact, score)`` for a query within a scope, honoring safe-by-default
    scoping and skipping superseded rows.
    """

    def __init__(self, inner, index, embedder):
        self._inner = inner
        self._index = index
        self._embedder = embedder
        self._indexed: set[int] = set()

    # -- FactStore delegation (+ indexing on add) --
    def add(self, fact: Fact) -> int:
        fid = self._inner.add(fact)
        if fid not in self._indexed:
            try:
                from fastaiagent.kb.chunking import Chunk

                emb = self._embedder.embed([fact.fact])[0]
                chunk = Chunk(
                    id=str(fid),
                    content=fact.fact,
                    metadata={
                        "scope": fact.scope,
                        "scope_id": fact.scope_id,
                        "project_id": fact.project_id,
                    },
                    index=0,
                    start_char=0,
                    end_char=len(fact.fact),
                )
                self._index.add([chunk], [emb])
                self._indexed.add(fid)
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "SemanticFactStore: failed to index fact %s", fid, exc_info=True
                )
        return fid

    def get(self, fact_id: int) -> Fact | None:
        return self._inner.get(fact_id)

    def list_active(self, scope, scope_id="", project_id="", limit=None):
        return self._inner.list_active(scope, scope_id, project_id, limit)

    def supersede(self, old_id: int, new_id: int) -> None:
        self._inner.supersede(old_id, new_id)

    def delete(self, scope, scope_id="", project_id="", fact=None) -> int:
        # Best-effort: drop matching vectors before the rows disappear.
        try:
            ids = [
                str(f.id)
                for f in self._inner.list_active(scope, scope_id, project_id)
                if fact is None or f.fact == fact
            ]
            if ids:
                self._index.delete(ids)
                self._indexed.difference_update(int(i) for i in ids)
        except Exception:
            pass
        return self._inner.delete(scope, scope_id, project_id, fact)

    # -- Semantic search --
    def search(
        self, query: str, scope: Scope, scope_id: str = "", project_id: str = "", top_k: int = 10
    ) -> list[tuple[Fact, float]]:
        """Return ``(Fact, score)`` for facts semantically matching ``query``."""
        if scope in ("user", "project") and scope_id == "":
            return []
        emb = self._embedder.embed([query])[0]
        hits = self._index.search(emb, max(top_k * 5, top_k))
        out: list[tuple[Fact, float]] = []
        for chunk, score in hits:
            m = chunk.metadata or {}
            if m.get("scope") != scope or m.get("project_id", "") != project_id:
                continue
            if scope_id and scope_id != "*" and m.get("scope_id") != scope_id:
                continue
            f = self._inner.get(int(chunk.id))
            if f is None or f.superseded_by is not None:
                continue
            out.append((f, float(score)))
            if len(out) >= top_k:
                break
        return out


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
"""


class PostgresFactStore:
    """``FactStore`` over Postgres (via ``psycopg`` v3). Requires the
    ``fastaiagent[postgres]`` extra. Table is created on first use."""

    def __init__(self, dsn: str):
        try:
            import psycopg  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "PostgresFactStore needs psycopg — install fastaiagent[postgres]"
            ) from e
        self._dsn = dsn
        with self._conn() as c:
            c.execute(_PG_DDL)
            c.commit()

    def _conn(self):
        import psycopg

        return psycopg.connect(self._dsn)

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
        )

    _COLS = (
        "id, scope, scope_id, fact, source_trace_id, "
        "confidence, created_at, superseded_by, project_id"
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
            cur = c.execute(
                "INSERT INTO learned_memory "
                "(scope, scope_id, fact, source_trace_id, confidence, created_at, project_id) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (
                    fact.scope,
                    fact.scope_id,
                    fact.fact,
                    fact.source_trace_id,
                    fact.confidence,
                    created,
                    fact.project_id,
                ),
            )
            new_id = int(cur.fetchone()[0])
            c.commit()
            return new_id

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


class RedisFactStore:
    """``FactStore`` over Redis. Requires the ``redis`` package.

    Layout: each fact is a hash ``fa:fact:{id}``; ids are minted from
    ``fa:fact:seq``; ``fa:uniq`` maps the idempotency tuple → id; active ids are
    tracked in per-``(scope,scope_id,project)`` sets (``fa:act:…``) with the set
    of scope_ids per ``(scope,project)`` in ``fa:sids:…`` so ``scope_id="*"`` /
    permissive ``agent`` reads can fan out.
    """

    def __init__(self, url: str, *, namespace: str = "fa"):
        try:
            import redis
        except ImportError as e:  # pragma: no cover
            raise ImportError("RedisFactStore needs the redis package: pip install redis") from e
        self._r = redis.from_url(url, decode_responses=True)
        self._ns = namespace

    def _k(self, *parts: str) -> str:
        return ":".join((self._ns, *parts))

    def _act_key(self, scope: str, scope_id: str, project_id: str) -> str:
        return self._k("act", scope, scope_id, project_id)

    def _all_key(self, scope: str, scope_id: str, project_id: str) -> str:
        return self._k("all", scope, scope_id, project_id)

    def _sids_key(self, scope: str, project_id: str) -> str:
        return self._k("sids", scope, project_id)

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
        self._r.hset(
            self._k("fact", str(fid)),
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
            },
        )
        self._r.set(uniq, fid)
        self._r.sadd(self._act_key(fact.scope, fact.scope_id, fact.project_id), fid)
        self._r.sadd(self._all_key(fact.scope, fact.scope_id, fact.project_id), fid)
        self._r.sadd(self._sids_key(fact.scope, fact.project_id), fact.scope_id)
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
        ids = self._ids("act", scope, scope_id, project_id)
        facts = [f for fid in ids if (f := self.get(int(fid)))]
        facts.sort(key=lambda f: f.created_at or 0.0, reverse=True)
        return facts[:limit] if limit is not None else facts

    def supersede(self, old_id: int, new_id: int) -> None:
        old = self.get(old_id)
        new = self.get(new_id)
        if not old or not new:
            raise ValueError(f"supersede: missing row(s) old_id={old_id} new_id={new_id}")
        self._r.hset(self._k("fact", str(old_id)), "superseded_by", new_id)
        self._r.srem(self._act_key(old.scope, old.scope_id, old.project_id), old_id)

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
            self._r.delete(self._k("uniq", self._uniq(f.scope, f.scope_id, f.fact, f.project_id)))
            n += 1
        return n
