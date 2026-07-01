"""External memory backends — the same `Memory` API over Postgres and Redis.

`Memory(location=...)` swaps the durable fact store without changing any agent
code. This script runs the full fact lifecycle (persist → retrieve → update /
supersede → forget) and a semantic query against whichever backends are
reachable — Postgres, Redis, and SQLite (always).

Start the servers (once):

    docker run -d --name fa-pg -e POSTGRES_PASSWORD=test -e POSTGRES_DB=fastaiagent_test \\
        -p 127.0.0.1:55432:5432 postgres:16-alpine
    docker run -d --name fa-redis -p 127.0.0.1:56379:6379 redis:7-alpine

Then:

    pip install 'fastaiagent[postgres,redis,kb]'
    PG_DSN=postgresql://postgres:test@127.0.0.1:55432/fastaiagent_test \\
    REDIS_URL=redis://127.0.0.1:56379/0 \\
        python companion.py

Unset backends are skipped with a hint. No LLM required.
"""

from __future__ import annotations

import os

from fastaiagent import Memory


def _reachable(location: str) -> bool:
    try:
        from fastaiagent.learn import make_fact_store

        make_fact_store(location)  # connects / creates table
        return True
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"  (skip {location.split('://')[0]}: {type(e).__name__}: {str(e)[:70]})")
        return False


def exercise(name: str, location) -> None:
    print(f"\n=== {name} ===")
    # semantic='auto' needs an embedder (fastaiagent[kb]); degrade if absent.
    try:
        mem = Memory(location=location, semantic="auto")
        semantic = True
    except Exception:
        mem = Memory(location=location)
        semantic = False

    # CREATE — global (shared truth) + per-user personalization
    mem.persist("Support replies within 24 hours.", tier="global")
    mem.persist("Alice prefers email over phone.", tier="user", id="alice")
    mem.persist("Alice is on the Pro plan.", tier="user", id="alice")
    print("  created:", [f.fact for f in mem.retrieve(tier="user", id="alice")])

    # UPDATE — supersede an old fact (history preserved)
    mem.update("Alice prefers Slack over email.", old="Alice prefers email over phone.",
               tier="user", id="alice")
    print("  after update:", [f.fact for f in mem.retrieve(tier="user", id="alice")])

    # ISOLATION — bob sees nothing of alice; empty id is safe
    print("  bob:", mem.retrieve(tier="user", id="bob"), " | empty-id (safe):",
          mem.retrieve(tier="user"))

    # SEMANTIC — retrieve by meaning (if an embedder is available)
    if semantic:
        hits = mem.retrieve("which plan is the user on?", tier="user", id="alice", limit=1)
        print("  semantic top hit:", [f.fact for f in hits])

    # REMOVE — forget the whole subject (incl. superseded history)
    removed = mem.forget(tier="user", id="alice")
    print("  removed:", removed, "-> now:", mem.retrieve(tier="user", id="alice"))


def main() -> int:
    import tempfile

    from fastaiagent.learn import MemoryStore

    # SQLite always runs.
    exercise("SQLite (default)", MemoryStore(db_path=tempfile.mktemp(suffix=".db")))

    pg = os.environ.get("PG_DSN")
    if pg and _reachable(pg):
        exercise("Postgres", pg)
    else:
        print("\n=== Postgres === (set PG_DSN to a running postgres to run this)")

    redis_url = os.environ.get("REDIS_URL")
    if redis_url and _reachable(redis_url):
        exercise("Redis", redis_url)
    else:
        print("\n=== Redis === (set REDIS_URL to a running redis to run this)")

    print(
        "\nNote: agent runs against any backend emit memory.read/write/persist/retrieve "
        "trace spans (browsable in `fastaiagent ui`). The UI Memory *page* browses the "
        "local SQLite store; external-backend facts are observed via the trace spans."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
