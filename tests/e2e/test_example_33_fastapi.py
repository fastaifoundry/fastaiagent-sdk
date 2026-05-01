"""End-to-end test for ``examples/33_deploy_fastapi.py``.

Imports the example's FastAPI ``app`` directly (the filename starts with
a digit, so a normal ``import examples.33_deploy_fastapi`` won't parse —
we use ``importlib.util.spec_from_file_location``). Then exercises
``/health`` and ``/run`` via ``fastapi.testclient.TestClient`` so the
lifespan runs correctly and the request goes through the same routing
code a deployed instance hits.

This catches the common failure modes: missing route, broken request /
response model, lifespan that crashes at startup, agent that doesn't
build under the deployed environment. It does NOT exercise
``uvicorn.run`` (port binding) — that's covered by the fact that
``uvicorn`` itself ships with a tested binding loop.

Marked ``e2e`` so it runs alongside the other live-LLM tests and out
of the fast unit matrix.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLE = REPO_ROOT / "examples" / "33_deploy_fastapi.py"

pytestmark = pytest.mark.e2e


def _load_example_app() -> Any:
    """Import the example file as an anonymous module and return ``.app``."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    spec = importlib.util.spec_from_file_location("_ex33_deploy_fastapi", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


def test_fastapi_example_health_endpoint() -> None:
    """``/health`` returns ``{"status": "ok"}`` without any LLM call.

    Runs even without an OpenAI key because ``/health`` doesn't touch
    the agent. Lifespan still runs (it builds the agent once at startup),
    which means ``LLMClient(...)`` must at least construct cleanly with
    just the key — but no LLM call is issued.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip(
            "OPENAI_API_KEY required — lifespan builds an LLMClient at startup"
        )
    from fastapi.testclient import TestClient

    app = _load_example_app()
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_fastapi_example_run_returns_real_llm_output() -> None:
    """``POST /run`` round-trips through the real OpenAI agent."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY required for the real /run round-trip")
    from fastapi.testclient import TestClient

    app = _load_example_app()
    with TestClient(app) as client:
        r = client.post("/run", json={"input": "What is 2 + 2? Answer with just the digit."})
        assert r.status_code == 200, r.text
        data = r.json()
        # Output is a string, latency is an int, tokens_used is an int.
        assert isinstance(data["output"], str) and data["output"].strip()
        assert "4" in data["output"]
        assert isinstance(data["latency_ms"], int)
        assert isinstance(data["tokens_used"], int) and data["tokens_used"] >= 0


def test_fastapi_example_run_rejects_empty_input() -> None:
    """The example explicitly raises 400 on empty input — verify it still does."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY required because lifespan builds an LLMClient")
    from fastapi.testclient import TestClient

    app = _load_example_app()
    with TestClient(app) as client:
        r = client.post("/run", json={"input": "   "})
        assert r.status_code == 400, r.text
