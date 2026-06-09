# Registered runner

A **registered runner** is a long-lived daemon you run **inside your own
boundary** (your network, with your tools / keys / KB). It connects out to the
platform, pulls the few jobs that must execute live — starting with the
**agent playground** — runs the real agent locally, and reports results back.
The platform never runs your agent code or holds your secrets; it only commands
the runner.

```bash
fastaiagent runner \
  --connect https://app.fastaiagent.net \
  --key $FASTAIAGENT_API_KEY \
  --labels region=eu --labels gpu=false \
  --max-concurrency 4
```

## How it works

```
register ──▶ heartbeat (every ttl/3) ──▶ long-poll commands ──▶ execute ──▶ report ──▶ ⟳
```

1. **Register** with your API key (`X-API-Key`) and advertise
   `capabilities:["live_playground"]`. The platform returns a `runner_id` and a
   short-lived **`runner_token`** (kept in memory only — never written to disk).
2. **Heartbeat** every `ttl_seconds / 3`, reporting `status` and the in-flight
   job count.
3. **Long-poll** for commands (≤30s). Each `live_playground` command carries the
   agent **config** (prompt, model, tool *specs*, KB refs) plus an input — never
   tools or keys.
4. **Execute** each job as its own asyncio task, bounded by `--max-concurrency`,
   inside a [`job_scope`](../durability/concurrency.md) so it binds *your* local
   tools / keys / KB. The job's trace is pushed through the normal
   `connect()` telemetry path.
5. **Report** the result (`completed` / `failed`, with the `trace_id`).

All calls after register authenticate with `Authorization: Bearer <runner_token>`.

## Resilience & shutdown

- **Re-register** automatically on auth loss (heartbeat miss / `401` / `404`),
  minting a fresh token — no credential at rest.
- **Backoff** (exponential, capped) on transient transport errors.
- **Graceful shutdown** on `Ctrl-C` / `SIGTERM`: stop pulling new work, **drain**
  in-flight jobs, then send a final `status="stopping"` heartbeat (the platform
  marks the runner offline and re-queues anything still in flight). There is no
  separate deregister endpoint.

## Scope

v1 handles the **`live_playground`** job type only; `eval_run` /
`guarded_live_rerun` / `tool_exec` are fast-follows. A `guarded_live_rerun` will
gate on a tool's [`replay_class`](../tools/index.md#replay-safety-replay_class)
and never execute a `side_effecting` tool.

The runner runs concurrent jobs for **one tenant** per process; per-job state is
request-scoped via [`job_scope`](../durability/concurrency.md), so each job must
run in its own asyncio task (the daemon guarantees this).
