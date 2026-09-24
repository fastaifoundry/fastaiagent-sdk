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

    **``E2E_PLATFORM_REQUIRED=1`` overrides the skip.** Without some way to
    demand this path it was an *unconditional* skip whenever ``E2E_SKIP_PLATFORM``
    was set, and the 2026-09-10 audit found what that cost:
    ``test_connected_guardrail_actions_e2e.py`` is the **only** place that pins
    severity/floor crossing the wire, a plane-authored mask/warn/override
    enforcing in-process, and a domain-wide rule producing an execution row — and
    all of it skipped on every PR, silently and by configuration rather than by
    accident. A skip is not a check; a gate with no way to demand it is not a gate.

    **It is deliberately not ``E2E_REQUIRED``**, which the first version of this
    reused and which broke the gate. That flag already means something else here:
    *"the core e2e gate must actually run — do not skip because a key is
    missing"* (see :func:`require_env`). CI sets ``E2E_REQUIRED=1`` and
    ``E2E_SKIP_PLATFORM=1`` **together**, on purpose — run the gate for real,
    without a platform — so overloading the first turned nine passing steps into
    hard failures. Two flags, two questions: *must the gate run at all* versus
    *must it include the platform round-trip*.
    """
    if not _skip_platform():
        return
    message = (
        "E2E_SKIP_PLATFORM=1 — platform-dependent step bypassed. "
        "Run locally without this flag (and with FASTAIAGENT_TARGET set) "
        "to exercise the platform push/verify path."
    )
    if os.environ.get("E2E_PLATFORM_REQUIRED") == "1":
        pytest.fail(
            f"{message} E2E_PLATFORM_REQUIRED=1 demands the platform path actually "
            "run — unset E2E_SKIP_PLATFORM, or drop E2E_PLATFORM_REQUIRED."
        )
    pytest.skip(message)


#: The persistent lab domain the connected gates author into (by name, when
#: ``E2E_PLANE_DOMAIN_ID`` is not set).
LAB_DOMAIN_NAME = "Guardrail Actions Lab"

#: One console session for the whole run — see :func:`plane_admin`.
_PLANE_ADMIN: tuple[Any, dict[str, str], str] | None = None


def plane_admin(target: str, *, purpose: str) -> tuple[Any, dict[str, str], str]:
    """A domain-admin console session on the lab domain: ``(client, headers, domain_id)``.

    **One login per test session.** The plane rate-limits logins, and each
    connected gate used to log in on its own, so running them together tripped
    ``Too many login attempts`` part-way through. ``client`` has
    ``base_url=target``.

    **The lab domain, never "the first one".** ``E2E_PLANE_DOMAIN_ID``, or the
    domain named :data:`LAB_DOMAIN_NAME`. Two gates used ``domains[0]``, which on
    the lab account is a different domain — they created projects and keys there
    and ran into its plan limits.

    Skips when the credentials are absent (``purpose`` says what they are for);
    **fails** when the plane refuses the login, because a gate that could not
    authenticate checked nothing.
    """
    global _PLANE_ADMIN
    if _PLANE_ADMIN is not None:
        return _PLANE_ADMIN

    import httpx

    email = os.environ.get("E2E_PLANE_EMAIL")
    password = os.environ.get("E2E_PLANE_PASSWORD")
    if not (email and password):
        pytest.skip(f"E2E_PLANE_EMAIL / E2E_PLANE_PASSWORD not set — {purpose}")

    client = httpx.Client(base_url=target, timeout=30)
    try:
        resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    except httpx.TransportError as exc:
        pytest.skip(f"plane unreachable at {target} ({exc}) — is the local plane running?")
    if resp.status_code != 200:
        pytest.fail(f"plane login refused: HTTP {resp.status_code} {resp.text[:160]}")
    headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}

    domain_id = os.environ.get("E2E_PLANE_DOMAIN_ID")
    if not domain_id:
        domains = client.get("/api/v1/users/me/domains", headers=headers).json()
        lab = next((d for d in domains if d["name"] == LAB_DOMAIN_NAME), None)
        if lab is None:
            pytest.skip(
                f"no '{LAB_DOMAIN_NAME}' domain on this plane — create it (or set "
                "E2E_PLANE_DOMAIN_ID) so the gate does not write into a shared domain."
            )
        domain_id = lab["id"]
    _PLANE_ADMIN = (client, headers, str(domain_id))
    return _PLANE_ADMIN


def require_lab_project(client: Any, headers: dict[str, str], domain_id: str) -> str:
    """The id of a project in the lab domain — reused, never created.

    ``E2E_PLANE_PROJECT_ID``, or the domain's first project. Gates used to create
    one per run, which runs into the plan's project cap (HTTP 402).
    """
    project_id = os.environ.get("E2E_PLANE_PROJECT_ID")
    if project_id:
        return project_id
    resp = client.get(f"/api/v1/domains/{domain_id}/projects", headers=headers)
    assert resp.status_code == 200, f"listing lab projects: {resp.status_code} {resp.text[:160]}"
    body = resp.json()
    projects = body if isinstance(body, list) else body.get("projects", body.get("items", []))
    if not projects:
        pytest.skip("the lab domain has no project — create one (or set E2E_PLANE_PROJECT_ID).")
    return str(projects[0]["id"])


def lab_guardrail_ids(client: Any, headers: dict[str, str], domain_id: str) -> set[str]:
    """Ids of the active, non-template guardrails in the lab domain."""
    resp = client.get("/api/v1/guardrails", headers=headers, params={"domain_id": domain_id})
    assert resp.status_code == 200, f"listing guardrails: {resp.status_code} {resp.text[:160]}"
    return {
        str(g["id"]) for g in resp.json() if not g.get("is_template") and g.get("is_active", True)
    }


def remove_guardrails_added_since(
    client: Any, headers: dict[str, str], domain_id: str, before: set[str]
) -> list[str]:
    """Remove every lab guardrail that is not in ``before``; returns their ids.

    A pushed agent's guardrails are installed on the plane, so a gate that pushes
    an agent carrying one leaves a rule behind that every later gate receives. A
    rule with recorded executions cannot be deleted (audit evidence), so it is
    deactivated instead.
    """
    added = sorted(lab_guardrail_ids(client, headers, domain_id) - before)
    for rule_id in added:
        gone = client.delete(f"/api/v1/guardrails/{rule_id}", headers=headers)
        if gone.status_code == 409:
            client.put(f"/api/v1/guardrails/{rule_id}", headers=headers, json={"is_active": False})
    return added


@pytest.fixture(autouse=True, scope="module")
def _fresh_connection_per_connected_module(request: pytest.FixtureRequest) -> Any:
    """Start and end every ``test_connected_*`` module disconnected, with fresh drains.

    ``connect()`` wires exporters that cache the ``local.db`` they drain on first
    use — right for a real process, which has one store for its lifetime. Run
    several connected gates in one pytest process, though, and a later module
    writes spans / HITL events to its own fresh store while the drain keeps
    reading the previous one: nothing ships, and the gate fails with "no execution
    row" or "never synced". Each passed on its own.
    """
    connected = request.module.__name__.rsplit(".", 1)[-1].startswith("test_connected_")
    if connected:
        from tests._governance_plane import reset_connection

        reset_connection()
    yield
    if connected:
        from tests._governance_plane import reset_connection

        reset_connection()


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
        pytest.skip(f"Ollama daemon at {host} returned status {resp.status_code}")


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
        pytest.skip(f"OTLP query API at {query_url} returned status {q.status_code}")


@pytest.fixture(scope="module")
def gate_state() -> dict[str, Any]:
    """Module-scoped scratchpad threading state across ordered gate sub-tests.

    Each step writes the artifacts it produces here (agent, trace_id, replay,
    forked, eval results). Downstream steps read them back. This gives per-step
    pytest granularity — a failure names the exact step that broke — while
    still running the pipeline end-to-end in a single file.
    """
    return {}
