"""Minimal uvicorn launcher for the foreign-OTel Playwright spec.

Boots ``build_app`` against a seeded DB with auth disabled and unscoped project
(so every seeded span renders). Deliberately loads no example runners — the
spec only reads trace pages.

Usage:  python scripts/_otel_ui_server.py --db <db_path> [--host H --port P]
"""

from __future__ import annotations

import argparse

import uvicorn

from fastaiagent.ui.server import build_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7846)
    args = parser.parse_args()

    app = build_app(db_path=args.db, no_auth=True, project_id="")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
