"""Per-job request-scoping for process-global SDK state (task 2.5).

A runner runs concurrent jobs for ONE tenant inside a single process. These
ContextVars let :func:`fastaiagent.runtime.job_scope` overlay the process-global
state — the connect() connection, the tool registry, and the trace-normalize
flags — so concurrent jobs don't clobber each other.

When no job scope is active (the common single-agent path) every consumer falls
back to its module global, so behavior is **byte-for-byte unchanged**.

ContextVars are async-task-local: a ``ContextVar`` copy is made per asyncio
task at creation. The registered-runner daemon (task 2.6) MUST therefore launch
each job as its own ``asyncio.create_task`` (not ``asyncio.gather`` of
coroutines in one task) for this isolation to hold; thread offloads need
``contextvars.copy_context().run(...)``.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

# Sentinel distinguishing "no per-job override" from a real value of None
# (e.g. ``framework=None`` is a legitimate scoped value).
UNSET: Any = object()

# A per-job ``_Connection`` (api_key/target/project). None -> use the global.
scoped_connection: ContextVar[Any] = ContextVar("fa_scoped_connection", default=None)

# A per-job tool dict overlaid on the global ToolRegistry (job wins on a name
# collision). None -> no scope; auto-registration falls through to the global.
scoped_tools: ContextVar[Any] = ContextVar("fa_scoped_tools", default=None)

# A per-job local project_id used to stamp spans. None -> use the global
# resolution (override / cached singleton / config.toml).
scoped_project_id: ContextVar[Any] = ContextVar("fa_scoped_project_id", default=None)

# Per-job trace-normalize flags. UNSET -> use the storage module globals.
scoped_normalize: ContextVar[Any] = ContextVar("fa_scoped_normalize", default=UNSET)
scoped_framework: ContextVar[Any] = ContextVar("fa_scoped_framework", default=UNSET)
