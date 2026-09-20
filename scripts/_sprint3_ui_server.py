"""Boot a Local UI server with Sprint 3 fixtures.

No registered runners — Sprint 3's three features (trace comparison,
dataset editor, richer filtering) are all read-side, so the server
only needs the seeded DB plus the on-disk datasets the seed script
laid down. Used by ``scripts/capture-sprint3-screenshots.sh``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastaiagent.ui.server import build_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7847)
    parser.add_argument("--project-id", default="sprint3-demo")
    args = parser.parse_args()

    app = build_app(
        db_path=args.db,
        no_auth=True,
        project_id=args.project_id,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
