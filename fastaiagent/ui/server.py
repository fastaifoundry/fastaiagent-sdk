"""FastAPI app factory for ``fastaiagent ui``.

The app is built in :func:`build_app` so the CLI, tests, and embedding
environments can all construct their own instance with their own DB path,
auth file, and ``--no-auth`` flag. ``uvicorn`` then serves it.
"""

from __future__ import annotations

import importlib.resources as resources
from pathlib import Path

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
    evals,
    guardrails,
    kb,
    overview,
    prompts,
    replay,
    traces,
    workflows,
)


def _static_dir() -> Path | None:
    """Locate the bundled frontend static assets, if present."""
    # First: packaged location inside the wheel (fastaiagent/ui/static).
    try:
        packaged = resources.files("fastaiagent.ui").joinpath("static")
        if packaged.is_dir():
            return Path(str(packaged))
    except (ModuleNotFoundError, AttributeError, FileNotFoundError):
        pass
    # Fall back to the sibling of this file (editable installs).
    candidate = Path(__file__).parent / "static"
    return candidate if candidate.exists() else None


def build_app(
    *,
    db_path: str | None = None,
    auth_path: Path | None = None,
    no_auth: bool = False,
) -> FastAPI:
    """Create a FastAPI app bound to a specific local.db and auth.json."""
    resolved_db = db_path or get_config().local_db_path
    resolved_auth = auth_path or default_auth_path()

    # Eagerly ensure the schema exists so every route can assume it's there.
    init_local_db(resolved_db).close()

    app = FastAPI(title="FastAIAgent", version="0.1", docs_url=None, redoc_url=None)
    app.state.context = AppContext(
        db_path=resolved_db,
        auth_path=resolved_auth,
        no_auth=no_auth,
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
    ):
        app.include_router(r)

    static = _static_dir()
    if static is not None:
        assets_dir = static / "assets"
        if assets_dir.exists():
            app.mount(
                "/assets", StaticFiles(directory=str(assets_dir)), name="assets"
            )

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
