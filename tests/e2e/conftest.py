"""Shared fixtures and helpers for the end-to-end quality gate."""

from __future__ import annotations

import os
from typing import Any

import pytest

# Env required regardless of mode — the LLM provider is always exercised.
CORE_ENV = ["OPENAI_API_KEY"]
# Env required only when the gate also exercises the platform push path.
PLATFORM_ENV = ["FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET"]


def _skip_platform() -> bool:
    """True when the gate should bypass platform-dependent steps.

    Set ``E2E_SKIP_PLATFORM=1`` on CI to run the gate without connecting to
    or verifying against a remote/local platform. Locally, leave it unset
    and point ``FASTAIAGENT_TARGET`` at your docker-compose platform.
    """
    return os.environ.get("E2E_SKIP_PLATFORM") == "1"


def require_env() -> None:
    """Skip the gate locally when secrets are absent; hard-fail in CI.

    CI sets ``E2E_REQUIRED=1``. Locally, developers get a clean skip so
    ``pytest tests/e2e/`` is not a permanent red mark on their machine.

    When ``E2E_SKIP_PLATFORM=1`` is set, only core env (OpenAI key) is
    required — the platform-dependent env vars are not demanded.
    """
    needed = list(CORE_ENV)
    if not _skip_platform():
        needed.extend(PLATFORM_ENV)

    missing = [k for k in needed if not os.environ.get(k)]
    if not missing:
        return
    message = f"Missing required env for e2e quality gate: {missing}"
    if os.environ.get("E2E_REQUIRED") == "1":
        pytest.fail(message)
    pytest.skip(message)


def require_platform() -> None:
    """Skip the current step when the gate is running in no-platform mode.

    Used on steps 2 (connect) and 10 (verify trace in dashboard) so CI runs
    without hitting a remote platform, while local runs against a
    docker-compose platform still exercise the full push/verify path.
    """
    if _skip_platform():
        pytest.skip(
            "E2E_SKIP_PLATFORM=1 — platform-dependent step bypassed. "
            "Run locally without this flag (and with FASTAIAGENT_TARGET set) "
            "to exercise the platform push/verify path."
        )


def require_anthropic() -> None:
    """Skip/fail the current test when ``ANTHROPIC_API_KEY`` is not set.

    Used by provider-specific gates (Anthropic, LangChain w/ Claude, etc.).
    Same skip-local, fail-on-CI contract as require_env().
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    message = "ANTHROPIC_API_KEY not set — skipping Anthropic-specific gate step"
    if os.environ.get("E2E_REQUIRED") == "1":
        pytest.fail(message)
    pytest.skip(message)


def require_import(module: str) -> None:
    """Skip the current test when an optional dependency is not importable.

    Used by integration gates (LangChain, CrewAI, etc.) that depend on
    packages listed under optional extras. Never fails in CI — missing
    optional deps are always a skip, even under E2E_REQUIRED=1, because
    CI explicitly installs ``[all,dev]`` and a missing import there is a
    packaging issue to fix separately, not a gate failure.
    """
    try:
        __import__(module)
    except ImportError:
        pytest.skip(f"Optional dependency '{module}' not importable — gate step skipped")


def require_ollama_running(host: str = "http://localhost:11434") -> None:
    """Skip the current test when a local Ollama daemon is not reachable.

    Always a skip (never a hard fail), even under ``E2E_REQUIRED=1``,
    because GitHub Actions runners do not have Ollama installed and
    most user laptops won't either. Locally, install + start Ollama
    (``brew install ollama && ollama serve``) and pull at least one
    small model (``ollama pull gemma2:2b``) to exercise this gate.
    """
    import httpx

    try:
        resp = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=2.0)
    except Exception as e:
        pytest.skip(f"Ollama daemon not reachable at {host}: {e}")
    if resp.status_code != 200:
        pytest.skip(
            f"Ollama daemon at {host} returned status {resp.status_code}"
        )


def require_otlp_endpoint(
    ingest_url: str = "http://localhost:4318/v1/traces",
    query_url: str = "http://localhost:16686/api/services",
) -> None:
    """Skip the current test when an OTLP collector is not reachable.

    Probes both the ingest URL (OTLP HTTP receiver) and the query URL
    (Jaeger / Tempo / similar query API). A GET against the ingest
    endpoint should return 405 Method Not Allowed (it's POST-only);
    a GET against the query URL should return 200. Both must be alive
    for the round-trip gate to mean anything.

    Always a skip (never a hard fail), even under ``E2E_REQUIRED=1``,
    because GitHub Actions runners do not have Jaeger running by
    default. Locally, ``docker run -p 4318:4318 -p 16686:16686 jaegertracing/all-in-one``
    (or equivalent) to exercise this gate.
    """
    import httpx

    try:
        resp = httpx.get(ingest_url, timeout=2.0)
    except Exception as e:
        pytest.skip(f"OTLP ingest not reachable at {ingest_url}: {e}")
    # OTLP HTTP ingest expects POST; 405 on GET is the healthy sign.
    if resp.status_code not in (200, 202, 405):
        pytest.skip(
            f"OTLP ingest at {ingest_url} returned unexpected status "
            f"{resp.status_code} on GET — collector may not be healthy"
        )

    try:
        q = httpx.get(query_url, timeout=2.0)
    except Exception as e:
        pytest.skip(f"OTLP query API not reachable at {query_url}: {e}")
    if q.status_code != 200:
        pytest.skip(
            f"OTLP query API at {query_url} returned status {q.status_code}"
        )


@pytest.fixture(scope="module")
def gate_state() -> dict[str, Any]:
    """Module-scoped scratchpad threading state across ordered gate sub-tests.

    Each step writes the artifacts it produces here (agent, trace_id, replay,
    forked, eval results). Downstream steps read them back. This gives per-step
    pytest granularity — a failure names the exact step that broke — while
    still running the pipeline end-to-end in a single file.
    """
    return {}
