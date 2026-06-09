# Concurrency & job scoping

The SDK's single-agent path uses a few **process-global** pieces of state — the
`connect()` connection, the tool registry, the local project id, and the
trace-normalize flags. That's exactly right for one agent in one process.

A **runner** is different: it runs *many jobs concurrently in one process* for
one tenant. `fa.job_scope(...)` request-scopes that global state per job so
concurrent jobs don't clobber each other.

```python
import fastaiagent as fa

async def run_job(cmd):
    with fa.job_scope(api_key=cmd.api_key, project=cmd.project, tools=cmd.tools):
        agent = build_agent(cmd)        # tools built here register job-locally
        return await agent.arun(cmd.input)
```

What a `job_scope` overlays (everything else is untouched):

| Scoped | Effect inside the scope |
|--------|-------------------------|
| connection (`api_key` / `target` / `project`) | `get_platform_api()` and platform reads use the job's connection; any field you omit inherits the global `connect()` |
| tool registry | lookups overlay the job's tools over the global registry (the job wins on a name collision); tools created inside the scope stay job-local |
| `project_id` | the job's spans are stamped with its project id |
| `normalize` / `framework` | per-job trace-normalize flags |

**Outside a `job_scope`** (the normal single-agent path) every accessor uses the
process global — behavior is byte-for-byte unchanged.

## The one rule: one asyncio task per job

`job_scope` is built on `ContextVar`s, which are **async-task-local** — a task
gets its own *copy* of the context when it is created. So a runner must launch
each job as its own task:

```python
# CORRECT — each job runs in its own task with an isolated context copy.
await asyncio.gather(*(asyncio.create_task(run_job(c)) for c in commands))
```

The isolation comes from that per-task context copy. The thing to avoid is
setting and awaiting *several* jobs' scopes within a **single shared task**
(e.g. a hand-rolled scheduler that interleaves coroutines in one task) — there
the scopes would overwrite each other. If a job offloads work to a thread, the
context does not propagate automatically; carry it across with
`contextvars.copy_context().run(...)`.

The shared OTel tracer/exporter is intentionally **not** per-job: one provider
pushes to the single tenant target, and each span carries its job's scoped
project id for attribution.

## See also

- [API reference](api-reference.md#job_scope) — the `job_scope` signature.
- `examples/71_job_scope.py` — a runnable, mock-free concurrency demo.
