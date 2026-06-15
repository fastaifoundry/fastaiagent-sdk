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
   `capabilities:["live_playground","eval_run"]`. The platform returns a
   `runner_id` and a short-lived **`runner_token`** (kept in memory only — never
   written to disk). On startup the runner also calls `connect()` with the **same
   key**, which wires the trace exporter so the jobs it runs push their traces to
   the platform. A rejected key **fails fast** with a clear error; an unreachable
   platform is tolerated (the runner still starts and traces buffer locally,
   draining when it returns).
2. **Heartbeat** every `ttl_seconds / 3`, reporting `status` and the in-flight
   job count.
3. **Long-poll** for commands (≤30s). A `live_playground` command carries the
   agent **config** (prompt, model, tool *specs*, KB refs) plus an input; an
   `eval_run` command carries the config plus a set of cases. Neither ever carries
   tools or keys.
4. **Execute** each job as its own asyncio task, bounded by `--max-concurrency`,
   inside a [`job_scope`](../durability/concurrency.md) so it binds *your* local
   tools / keys / KB. Each job's trace is pushed to the platform via the exporter
   wired at startup; the platform routes it to **your API key's project** (an
   `eval_run` runs the agent once per case and emits one trace per case).
5. **Report** the result (`completed` / `failed`, with the `trace_id` that links
   to the pushed trace). An `eval_run` reports the per-case outputs
   (`{"outputs":[{case_id, output, trace_id}, …]}`); the platform scores them.

All calls after register authenticate with `Authorization: Bearer <runner_token>`.

## Verifying traces reach the platform

Every job a runner executes pushes its trace to the platform, routed by your API
key. The example `examples/83_runner_trace_to_platform.py` runs one
`live_playground` job exactly the way the daemon does, then reads the trace back
from the plane. A real run against a local plane on `:20001`:

```text
============================================================
  Runner trace -> platform (Task A)
============================================================
  Connected: True  domain=8ccd14b5-…  project=ab4d5161-…
  Executing live_playground 'demo-1' (real gpt-4o-mini)…
  status=completed  output='OK'  trace_id=19c1f4083a0409d075d8d18d7a7c3871
  Flushed exporter -> POST /public/v1/traces/ingest

  Verifying on the platform (GET /public/v1/traces/{id})…
  ✓ trace 19c1f4083a0409d075d8d18d7a7c3871  source=sdk  status=completed  spans=[agent.demo, llm.openai.gpt-4o-mini]
============================================================
```

The `trace_id` the runner reports in `POST /runners/{id}/results` is the **same
id** the trace lands under (`source=sdk`), so the console links each job result to
its full trace — here the agent span and its LLM call. An `eval_run` emits one such
trace per case.

## Resilience & shutdown

- **Re-register** automatically on auth loss (heartbeat miss / `401` / `404`),
  minting a fresh token — no credential at rest.
- **Backoff** (exponential, capped) on transient transport errors.
- **Graceful shutdown** on `Ctrl-C` / `SIGTERM`: stop pulling new work, **drain**
  in-flight jobs, then send a final `status="stopping"` heartbeat (the platform
  marks the runner offline and re-queues anything still in flight). There is no
  separate deregister endpoint.

## Scope

The runner handles the **`live_playground`** and **`eval_run`** job types;
`guarded_live_rerun` / `tool_exec` are fast-follows. A `guarded_live_rerun` will
gate on a tool's [`replay_class`](../tools/index.md#replay-safety-replay_class)
and never execute a `side_effecting` tool. For an `eval_run` the runner only
**executes** each case and returns the outputs — the platform applies the suite's
scorers / criteria centrally.

The runner runs concurrent jobs for **one tenant** per process; per-job state is
request-scoped via [`job_scope`](../durability/concurrency.md), so each job must
run in its own asyncio task (the daemon guarantees this). Because a runner is
single-tenant and the platform routes traces by your API key, all its job traces
land in that key's project — the local trace store is just a buffer.
