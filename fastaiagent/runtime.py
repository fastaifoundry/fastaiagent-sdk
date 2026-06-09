"""Public runtime-scoping API.

:func:`job_scope` lets a runner (or any host running concurrent jobs in one
process) overlay the SDK's process-global state *per job* — the ``connect()``
connection, the tool registry, the local ``project_id``, and the trace-normalize
flags — without changing the common single-agent path. See
:mod:`fastaiagent._internal.scope` for the mechanism and the per-asyncio-task
isolation rule the registered-runner daemon (task 2.6) must honor.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from fastaiagent._internal import scope as _scope


@contextmanager
def job_scope(
    *,
    api_key: str | None = None,
    target: str | None = None,
    project: str | None = None,
    tools: Sequence[Any] | None = None,
    normalize: bool | None = None,
    framework: str | None = None,
) -> Iterator[None]:
    """Scope process-global SDK state to the current job (and asyncio task).

    Wrap each concurrent job a runner executes so its connection, tool registry,
    local ``project_id`` and trace-normalize flags don't clobber sibling jobs.
    **Outside** a ``job_scope`` (the common single-agent path) every accessor
    uses the process global, so behavior is unchanged.

    Args:
        api_key / target / project: a per-job platform connection + project.
            Any field omitted inherits the global ``connect()`` connection.
        tools: per-job tools. Lookups overlay these over the global registry
            (the job wins on a name collision), and tools auto-registered inside
            the scope stay job-local.
        normalize / framework: per-job trace-normalize flags.

    ContextVar-based and async-task-local: a job MUST run in its own asyncio
    task for isolation to hold (see the runner daemon, task 2.6). ``with`` is
    synchronous because it only sets/resets ContextVars; the platform client is
    created lazily on first use.
    """
    tokens: list[tuple[Any, Any]] = []

    if api_key is not None or target is not None or project is not None:
        from fastaiagent.client import _Connection, _global_connection, _normalize_target

        conn = _Connection()
        # Inherit the global tenant for any field the caller didn't override.
        conn.api_key = api_key if api_key is not None else _global_connection.api_key
        conn.target = (
            _normalize_target(target) if target is not None else _global_connection.target
        )
        conn.project = project if project is not None else _global_connection.project
        tokens.append((_scope.scoped_connection, _scope.scoped_connection.set(conn)))

    # Always isolate the tool registry within a scope so tools created/registered
    # inside the job stay job-local instead of leaking to the global registry.
    job_tools: dict[str, Any] = {t.name: t for t in tools} if tools else {}
    tokens.append((_scope.scoped_tools, _scope.scoped_tools.set(job_tools)))

    if project is not None:
        # Stamp the job's spans with its project id (read in-task at span end).
        tokens.append((_scope.scoped_project_id, _scope.scoped_project_id.set(project)))
    if normalize is not None:
        tokens.append((_scope.scoped_normalize, _scope.scoped_normalize.set(normalize)))
    if framework is not None:
        tokens.append((_scope.scoped_framework, _scope.scoped_framework.set(framework)))

    try:
        yield
    finally:
        for var, tok in reversed(tokens):
            var.reset(tok)
