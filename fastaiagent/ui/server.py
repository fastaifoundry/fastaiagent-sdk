"""FastAPI app factory for ``fastaiagent ui``.

The app is built in :func:`build_app` so the CLI, tests, and embedding
environments can all construct their own instance with their own DB path,
auth file, and ``--no-auth`` flag. ``uvicorn`` then serves it.
"""

from __future__ import annotations

import importlib.resources as resources
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from fastaiagent._internal.config import get_config
from fastaiagent.ui.auth import default_auth_path
from fastaiagent.ui.db import init_local_db
from fastaiagent.ui.deps import AppContext
from fastaiagent.ui.routes import (
    agents,
    analytics,
    auth,
    datasets,
    evals,
    executions,
    filter_presets,
    guardrails,
    kb,
    overview,
    playground,
    prompts,
    replay,
    traces,
    workflows,
)

logger = logging.getLogger(__name__)


def _static_dir() -> Path | None:
    """Locate the bundled frontend static assets, if present."""
    # First: packaged location inside the wheel (fastaiagent/ui/static).
    try:
        packaged = resources.files("fastaiagent.ui").joinpath("static")
        if packaged.is_dir():
            return Path(str(packaged))
    except (ModuleNotFoundError, AttributeError, FileNotFoundError):
        logger.debug("Packaged static assets not found, falling back to sibling dir", exc_info=True)
    # Fall back to the sibling of this file (editable installs).
    candidate = Path(__file__).parent / "static"
    return candidate if candidate.exists() else None


def build_app(
    *,
    db_path: str | None = None,
    auth_path: Path | None = None,
    no_auth: bool = False,
    runners: Iterable[Any] | None = None,
    project_id: str | None = None,
) -> FastAPI:
    """Create a FastAPI app bound to a specific local.db and auth.json.

    ``runners`` (optional) is an iterable of resumable objects (Chain,
    Agent, Swarm, Supervisor) the server can call ``aresume(...)`` on.
    Each must expose ``.name`` and ``.aresume(...)``. The
    ``POST /api/executions/{id}/resume`` endpoint looks one up by the
    checkpoint's ``chain_name`` field and returns 503 if no match.

    ``project_id`` (optional) overrides the project the UI scopes to.
    When omitted, ``ProjectConfig.get_project_id()`` is used (which
    resolves ``./.fastaiagent/config.toml`` or the directory name).
    Endpoints filter SQL by this id so multiple projects can share the
    same DB (Postgres) without cross-contamination.
    """
    resolved_db = db_path or get_config().local_db_path
    resolved_auth = auth_path or default_auth_path()
    # Default to unscoped (project_id="") so test fixtures that don't seed
    # project_id keep working. The ``fastaiagent ui`` CLI explicitly sets
    # this via ProjectConfig so real users get isolation by default.
    resolved_project_id = project_id if project_id is not None else ""

    # Eagerly ensure the schema exists so every route can assume it's there.
    init_local_db(resolved_db).close()

    runner_map: dict[str, Any] = {}
    if runners is not None:
        for r in runners:
            name = getattr(r, "name", None)
            if not name or not hasattr(r, "aresume"):
                raise ValueError(
                    "build_app(runners=...) entries must have a .name and "
                    "an .aresume() method (Chain/Agent/Swarm/Supervisor). "
                    f"Got: {r!r}"
                )
            runner_map[str(name)] = r

    app = FastAPI(title="FastAIAgent", version="0.1", docs_url=None, redoc_url=None)
    app.state.context = AppContext(
        db_path=resolved_db,
        auth_path=resolved_auth,
        no_auth=no_auth,
        runners=runner_map,
        project_id=resolved_project_id,
    )

    for r in (
        auth.router,
        overview.router,
        traces.router,
        replay.router,
        evals.router,
        prompts.router,
        guardrails.router,
        agents.router,
        analytics.router,
        kb.router,
        workflows.router,
        executions.router,
        playground.router,
        datasets.router,
        filter_presets.router,
    ):
        app.include_router(r)

    static = _static_dir()
    if static is not None:
        assets_dir = static / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        index_file = static / "index.html"

        @app.get("/{path:path}", include_in_schema=False, response_model=None)
        async def spa_fallback(request: Request, path: str) -> FileResponse | JSONResponse:
            if path.startswith("api/"):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            candidate = static / path
            if candidate.is_file():
                return FileResponse(candidate)
            if index_file.exists():
                return FileResponse(index_file)
            return JSONResponse(
                {
                    "detail": (
                        "Frontend bundle not found — the Python wheel was "
                        "built without it. Run `cd ui-frontend && pnpm build` "
                        "from source."
                    )
                },
                status_code=503,
            )

    return app


__all__ = ["build_app"]
