"""Boot a Local UI server with Sprint 1 fixtures + a registered chain.

Used by ``scripts/capture-sprint1-screenshots.sh``. Imports the chain
from ``examples/47_workflow_topology.py`` so the topology screenshot
shows the same graph documented in that example.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastaiagent.ui.server import build_app  # noqa: E402


def _load_example_chain() -> object:
    """Import build_chain() from examples/47 without requiring the example
    package to be on sys.path."""
    path = REPO_ROOT / "examples" / "47_workflow_topology.py"
    spec = importlib.util.spec_from_file_location("ex47", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_chain()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7843)
    args = parser.parse_args()

    chain = _load_example_chain()
    app = build_app(db_path=args.db, no_auth=True, runners=[chain])
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
