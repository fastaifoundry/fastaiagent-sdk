# External memory backends (Postgres & Redis)

The same `Memory` API, a different durable store — swap `location=` with zero
agent-code changes:

```python
Memory(location="sqlite")                             # default (single-node)
Memory(location="postgres://user:pw@host:5432/db")    # fastaiagent[postgres]
Memory(location="redis://host:6379/0")                # fastaiagent[redis]
Memory(location=my_store)                             # any FactStore instance
```

`companion.py` runs the full fact lifecycle — **create → retrieve → update
(supersede) → forget** — plus a semantic query and per-user isolation, against
whichever backends are reachable (SQLite always).

## Run

```sh
docker run -d --name fa-pg -e POSTGRES_PASSWORD=test -e POSTGRES_DB=fastaiagent_test \
    -p 127.0.0.1:55432:5432 postgres:16-alpine
docker run -d --name fa-redis -p 127.0.0.1:56379:6379 redis:7-alpine

pip install 'fastaiagent[postgres,redis,kb]'
PG_DSN=postgresql://postgres:test@127.0.0.1:55432/fastaiagent_test \
REDIS_URL=redis://127.0.0.1:56379/0 \
    python companion.py
```

All three backends implement the same `FactStore` contract — idempotent add,
safe-by-default scoping (empty `user`/`project` id ⇒ nothing; `"*"` ⇒ all),
supersede (versioned, never overwrite), and guarded delete.

## Observability note
Agent runs against **any** backend emit `memory.read` / `memory.write` /
`memory.persist` / `memory.retrieve` trace spans, browsable in `fastaiagent ui`.
The UI **Memory page** browses the local SQLite store; facts written to an
external backend are observed via those trace spans (a shared UI over external
backends is future work).
