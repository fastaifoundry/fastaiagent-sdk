"""End-to-end quality gate — plane-authored guardrail actions (wire v1.9, 1.57.0).

No mocks. A real SDK connects to the LIVE local plane, pulls guardrails an
operator authored in the console over ``GET /public/v1/policy``, and asserts that
each one now does **what it says** inside the customer's own process rather than
turning every failure into a block:

    action=mask       -> the reply arrives redacted, not blocked
    action=warn       -> the failure is recorded, the run continues
    action=override   -> the operator's copy replaces the reply
    action=block      -> unchanged, and what every pre-v1.9 rule still does
    content_safety    -> blocks, naming the hazard category that tripped
    groundedness      -> passes with context, fails closed without it
    floor / severity  -> survive the wire onto the reconstructed Guardrail

The rules are created through the console API and torn down afterwards, so the
gate is repeatable. They are created **domain-wide** (no agent attachment) on
purpose: attaching a domain-wide rule to one agent narrows it to that agent and
silently removes it from every other one.

Gated by the ``connected_state_plane`` bundle flag: a 403 on the probe => a clean
skip with a setup message instead of an opaque failure.

Setup (once, per the lab convention of one persistent seeded domain per feature):

    E2E_PLANE_EMAIL / E2E_PLANE_PASSWORD  a domain-admin on the local plane
    FASTAIAGENT_TARGET                    http://localhost:20001
    FASTAIAGENT_API_KEY                   a key minted in that domain

Run:

    zsh -lc 'FASTAIAGENT_TARGET=http://localhost:20001 FASTAIAGENT_API_KEY=fa_k_... \
      E2E_PLANE_EMAIL=... E2E_PLANE_PASSWORD=... \
      .venv/bin/python -m pytest tests/e2e/test_connected_guardrail_actions_e2e.py -v -m e2e'
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from tests.e2e.conftest import require_env, require_platform

pytestmark = pytest.mark.e2e

SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"
LEAK = "The customer's SSN is 123-45-6789 on file."

#: Every rule this gate creates is suffixed with this, so a run never shares a
#: name with one a previous run left behind. That matters: the plane resolves a
#: guardrail span to its row **by name**, and where two rules in a domain share
#: one it picks the lower id — the older rule — so a leftover would silently
#: absorb this run's evidence. Rules that recorded executions cannot be deleted
#: (they are audit evidence), only deactivated, so leftovers are normal.
RUN = uuid.uuid4().hex[:8]


#: One login for the whole module. The plane rate-limits login attempts (rightly),
#: and a per-test login would trip it well before the gate finished.
_SESSION: tuple[Any, dict[str, str], str] | None = None


def _admin_session(target: str) -> tuple[Any, dict[str, str], str]:
    """Log in as a domain admin so the gate can author rules the way an operator does."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    import httpx

    email = os.environ.get("E2E_PLANE_EMAIL")
    password = os.environ.get("E2E_PLANE_PASSWORD")
    if not (email and password):
        pytest.skip(
            "E2E_PLANE_EMAIL / E2E_PLANE_PASSWORD not set — this gate authors "
            "guardrails through the console API, which needs a domain admin."
        )

    client = httpx.Client(base_url=target, timeout=30)
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    if resp.status_code != 200:
        pytest.skip(f"plane login failed ({resp.status_code}) — is the local plane running?")
    headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}

    domain_id = os.environ.get("E2E_PLANE_DOMAIN_ID")
    if not domain_id:
        domains = client.get("/api/v1/users/me/domains", headers=headers).json()
        lab = next((d for d in domains if d["name"] == "Guardrail Actions Lab"), None)
        if lab is None:
            pytest.skip(
                "no 'Guardrail Actions Lab' domain on this plane — create it (or set "
                "E2E_PLANE_DOMAIN_ID) so the gate does not author rules into a shared domain."
            )
        domain_id = lab["id"]
    _SESSION = (client, headers, domain_id)
    return _SESSION


class _Rules:
    """Create guardrails on the plane and clean them up afterwards."""

    def __init__(self, client: Any, headers: dict[str, str], domain_id: str) -> None:
        self._c, self._h, self._domain = client, headers, domain_id
        self.created: list[str] = []

    @property
    def client(self) -> Any:
        """The authoring session, for the assertions that read the plane back."""
        return self._c

    @property
    def headers(self) -> dict[str, str]:
        return self._h

    @property
    def domain_id(self) -> str:
        return self._domain

    def add(self, **body: Any) -> dict[str, Any]:
        body["name"] = f"{body['name']}-{RUN}"
        payload = {
            "description": "SDK guardrail-action e2e (safe to delete)",
            "guardrail_type": "output",
            "validation_mode": "blocking",
            "is_active": True,
            "on_error": "block",
            **body,
        }
        resp = self._c.post(
            "/api/v1/guardrails",
            headers=self._h,
            params={"domain_id": self._domain},
            json=payload,
        )
        assert resp.status_code < 300, f"authoring {body.get('name')!r} failed: {resp.text}"
        rule = resp.json()
        self.created.append(rule["id"])
        return rule

    def add_in_domain(self, domain_id: str, **body: Any) -> dict[str, Any]:
        """Author a rule in *another* domain, to prove tenancy holds."""
        body.setdefault("name", "unnamed")
        payload = {
            "description": "SDK guardrail-action e2e (safe to delete)",
            "guardrail_type": "output",
            "validation_mode": "blocking",
            "is_active": True,
            "on_error": "block",
            **body,
        }
        resp = self._c.post(
            "/api/v1/guardrails", headers=self._h, params={"domain_id": domain_id}, json=payload
        )
        assert resp.status_code < 300, f"authoring {body.get('name')!r} failed: {resp.text}"
        rule = resp.json()
        self.created.append(rule["id"])
        return rule

    def cleanup(self) -> None:
        for rule_id in self.created:
            resp = self._c.delete(f"/api/v1/guardrails/{rule_id}", headers=self._h)
            if resp.status_code == 409:
                # The rule has recorded executions, which are audit evidence the
                # plane deliberately refuses to discard. Deactivating takes it
                # off /policy, which is all the next test needs.
                self._c.put(
                    f"/api/v1/guardrails/{rule_id}", headers=self._h, json={"is_active": False}
                )
        self.created.clear()


@pytest.fixture(scope="module")
def module_local_db(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """One local store for the whole module, not one per test.

    The usual ``isolated_local_db`` gives every test a fresh SQLite file. That is
    right almost everywhere, but wrong here: ``connect()`` registers the platform
    exporter once, and the exporter caches its ``TraceStore`` on first use
    (``platform_export.PlatformSpanExporter._get_store``). Repointing the DB
    mid-process therefore leaves the exporter draining a file nothing writes to
    any more, and every span after the first test silently never ships.

    Caching the store is correct in production — a real process has one
    ``local.db`` for its lifetime — so the fixture matches reality instead of the
    SDK working around a test.
    """
    from fastaiagent._internal import instance as _instance
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config

    db_path = tmp_path_factory.mktemp("guardrail-actions-e2e") / "local.db"
    previous = os.environ.get("FASTAIAGENT_LOCAL_DB")
    os.environ["FASTAIAGENT_LOCAL_DB"] = str(db_path)
    reset_config()
    _project.set_project_id("test-proj")
    _instance.reset_for_testing()
    try:
        yield db_path
    finally:
        _instance.reset_for_testing()
        _project.reset_for_testing()
        if previous is None:
            os.environ.pop("FASTAIAGENT_LOCAL_DB", None)
        else:
            os.environ["FASTAIAGENT_LOCAL_DB"] = previous
        reset_config()


@pytest.fixture()
def plane(module_local_db: Any) -> Iterator[_Rules]:
    """A connected SDK plus an authoring session, torn down cleanly."""
    require_env()
    require_platform()

    import httpx

    from fastaiagent.client import _connection

    target = os.environ["FASTAIAGENT_TARGET"]
    client, headers, domain_id = _admin_session(target)

    # Feature-gate pre-check — skip cleanly on 403 rather than failing opaquely.
    probe = httpx.get(
        f"{target}/public/v1/policy",
        headers={"X-API-Key": os.environ["FASTAIAGENT_API_KEY"]},
        timeout=30,
    )
    if probe.status_code == 403:
        pytest.skip(
            "connected_state_plane not enabled for this domain — enable the bundle "
            "flag (or a per-subscription override) to run this gate."
        )
    assert probe.status_code == 200, f"/public/v1/policy returned {probe.status_code}"

    # A stray domain-wide rule left over from a previous run would enforce on
    # every assertion below and silently invalidate the gate — a masked reply
    # where the test expected an override reads as a product bug. Fail loudly.
    live = probe.json().get("guardrail_rules") or []
    if live:
        pytest.fail(
            "the lab domain already has active domain-wide guardrails "
            f"{[r['name'] for r in live]} — they would enforce on every step here. "
            "Deactivate or delete them, then re-run."
        )

    rules = _Rules(client, headers, domain_id)
    try:
        yield rules
    finally:
        rules.cleanup()
        # Drop the cached policy so the next test does not enforce this one's
        # rules against its own expectations.
        _connection.policy_cache = None


def _connect_and_refresh() -> None:
    """Connect (or re-pull) so the SDK sees rules authored a moment ago."""
    import fastaiagent as fa
    from fastaiagent.client import _connection

    if not _connection.is_connected:
        fa.connect(
            api_key=os.environ["FASTAIAGENT_API_KEY"],
            target=os.environ["FASTAIAGENT_TARGET"],
        )
    fa.refresh_policy()


#: A second domain, so the tenancy assertion has something to be excluded from.
_FOREIGN_DOMAIN_NAME = "Guardrail Actions Lab (foreign)"


def _ensure_foreign_domain(client: Any, headers: dict[str, str]) -> str | None:
    """Find or create a second domain. ``None`` when this account cannot make one."""
    domains = client.get("/api/v1/users/me/domains", headers=headers).json()
    existing = next((d for d in domains if d["name"] == _FOREIGN_DOMAIN_NAME), None)
    if existing:
        return str(existing["id"])
    resp = client.post(
        "/api/v1/domains",
        headers=headers,
        json={
            "name": _FOREIGN_DOMAIN_NAME,
            "description": "Tenancy control for the SDK guardrail-action e2e.",
        },
    )
    return str(resp.json()["id"]) if resp.status_code < 300 else None


def _executions(client: Any, headers: dict[str, str], guardrail_id: str) -> list[dict[str, Any]]:
    """Read back the rows the plane recorded for one rule.

    ``project_id`` is required by the endpoint and is the project the SDK
    connected as — the same one the ingest path stamps on every row.
    """
    from fastaiagent.client import _connection

    resp = client.get(
        "/api/v1/guardrail-executions",
        headers=headers,
        params={
            "guardrail_id": guardrail_id,
            "project_id": _connection.project_id,
            "limit": 100,
        },
    )
    assert resp.status_code == 200, f"GET /guardrail-executions -> {resp.status_code}: {resp.text}"
    rows = resp.json()
    return rows if isinstance(rows, list) else rows.get("items", rows)


def _flush_and_await_rows(
    client: Any,
    headers: dict[str, str],
    guardrail_id: str,
    *,
    expected: int = 1,
    timeout: float = 45.0,
) -> list[dict[str, Any]]:
    """Ship the buffered spans and wait for the plane to materialise the rows.

    The SDK exporter batches, and the plane resolves spans to
    ``guardrail_executions`` on ingest, so this is genuinely asynchronous — poll
    rather than sleep-and-hope.
    """
    import time

    from fastaiagent.trace.otel import get_tracer_provider

    get_tracer_provider().force_flush(15_000)
    deadline = time.monotonic() + timeout
    rows: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        rows = _executions(client, headers, guardrail_id)
        if len(rows) >= expected:
            return rows
        time.sleep(1.0)
        get_tracer_provider().force_flush(5_000)
    return rows


def _agent(response: str | list[str], name: str = "e2e-support") -> Any:
    """An agent with no local guardrails — everything it enforces came from the plane."""
    from fastaiagent import Agent
    from fastaiagent.testing.models import TestModel

    return Agent(name=name, llm=TestModel(response=response))


# --------------------------------------------------------------------------- #
# 1. The wire
# --------------------------------------------------------------------------- #
def test_every_rule_on_the_wire_carries_action_severity_and_floor(plane: _Rules) -> None:
    from fastaiagent.client import _connection
    from fastaiagent.guardrail.from_policy import guardrail_from_policy_rule

    created = plane.add(
        name="e2e-wire-check",
        implementation_type="regex",
        config={"pattern": SSN_PATTERN, "should_match": False},
        tripwire_message="SSN in output.",
        action="warn",
        severity="high",
    )
    _connect_and_refresh()

    rules = _connection.policy_cache["guardrail_rules"]
    assert rules, "the plane returned no guardrail rules"
    for rule in rules:
        assert "action" in rule and rule["action"] in (
            "block",
            "warn",
            "mask",
            "override",
            "reask",
        ), rule
        assert "severity" in rule
        assert isinstance(rule["floor"], bool)

    mine = next(r for r in rules if r["name"] == created["name"])
    built = guardrail_from_policy_rule(mine)
    assert (built.action, built.severity, built.floor) == ("warn", "high", False)

    # An SDK that predates v1.9 sees a rule with none of these keys. It must
    # behave exactly as it did before, which is to block.
    legacy = {k: v for k, v in mine.items() if k not in ("action", "severity", "floor")}
    assert guardrail_from_policy_rule(legacy).action == "block"


# --------------------------------------------------------------------------- #
# 2. mask — the headline: redacted, not blocked
# --------------------------------------------------------------------------- #
def test_a_console_authored_mask_rule_redacts_the_reply_in_process(plane: _Rules) -> None:
    plane.add(
        name="e2e-mask-pii",
        implementation_type="regex",
        config={
            "pattern": SSN_PATTERN,
            "should_match": False,
            "mask_token": "[REDACTED]",
        },
        tripwire_message="Output contained an SSN.",
        action="mask",
        severity="medium",
    )
    _connect_and_refresh()

    result = _agent(LEAK).run("what is on file?")
    assert "123-45-6789" not in result.output, "the SSN survived a mask rule"
    assert "[REDACTED]" in result.output
    assert result.output == "The customer's SSN is [REDACTED] on file."


def test_the_mask_lands_a_filtered_event_with_a_before_after_diff(plane: _Rules) -> None:
    from fastaiagent._internal.config import get_config
    from fastaiagent._internal.storage import SQLiteHelper

    rule = plane.add(
        name="e2e-mask-ui",
        implementation_type="regex",
        config={"pattern": SSN_PATTERN, "should_match": False, "mask_token": "[REDACTED]"},
        tripwire_message="Output contained an SSN.",
        action="mask",
        severity="critical",
    )
    _connect_and_refresh()

    get_config().ui_enabled = True
    _agent(LEAK).run("what is on file?")

    with SQLiteHelper(get_config().local_db_path) as db:
        rows = db.fetchall(
            "SELECT * FROM guardrail_events WHERE guardrail_name = ? ORDER BY timestamp DESC",
            (rule["name"],),
        )
    assert rows, "the masked run wrote no guardrail event"
    row = rows[0]
    assert row["outcome"] == "filtered"
    assert (row["action"], row["action_taken"]) == ("mask", "masked")
    assert row["severity"] == "critical"

    import json as _json

    metadata = _json.loads(row["metadata"] or "{}")
    assert "123-45-6789" in metadata["before"]
    assert "[REDACTED]" in metadata["after"]


# --------------------------------------------------------------------------- #
# 3. The other actions
# --------------------------------------------------------------------------- #
def test_a_warn_rule_records_the_failure_without_stopping_the_run(plane: _Rules) -> None:
    plane.add(
        name="e2e-warn",
        implementation_type="regex",
        config={"pattern": SSN_PATTERN, "should_match": False},
        tripwire_message="SSN in output.",
        action="warn",
    )
    _connect_and_refresh()
    assert _agent(LEAK).run("what is on file?").output == LEAK


def test_an_override_rule_substitutes_the_operators_copy(plane: _Rules) -> None:
    plane.add(
        name="e2e-override",
        implementation_type="regex",
        config={
            "pattern": SSN_PATTERN,
            "should_match": False,
            "override_message": "I cannot share account identifiers.",
        },
        tripwire_message="SSN in output.",
        action="override",
    )
    _connect_and_refresh()
    assert _agent(LEAK).run("what is on file?").output == "I cannot share account identifiers."


def test_a_block_rule_is_unchanged(plane: _Rules) -> None:
    from fastaiagent._internal.errors import GuardrailBlockedError

    plane.add(
        name="e2e-block",
        implementation_type="regex",
        config={"pattern": SSN_PATTERN, "should_match": False},
        tripwire_message="SSN in output.",
        action="block",
    )
    _connect_and_refresh()
    with pytest.raises(GuardrailBlockedError):
        _agent(LEAK).run("what is on file?")


# --------------------------------------------------------------------------- #
# 4. The two model-backed types (real judge — needs OPENAI_API_KEY)
# --------------------------------------------------------------------------- #
def test_a_content_safety_rule_blocks_and_names_the_category_that_tripped(
    plane: _Rules,
) -> None:
    from fastaiagent._internal.errors import GuardrailBlockedError
    from fastaiagent.guardrail.from_policy import plane_guardrails_for_agent

    rule = plane.add(
        name="e2e-content-safety",
        implementation_type="content_safety",
        config={"categories": ["S10", "S11"], "threshold": 0.5},
        tripwire_message="Content safety policy.",
        action="block",
        severity="high",
    )
    _connect_and_refresh()

    hateful = "People from that country are subhuman vermin and deserve nothing."
    with pytest.raises(GuardrailBlockedError):
        _agent(hateful).run("say something")

    # The verdict carries the per-category detail the console renders.
    (rail,) = [g for g in plane_guardrails_for_agent(None) if g.name == rule["name"]]
    verdict = rail.execute(hateful)
    assert verdict.passed is False
    assert verdict.metadata["taxonomy"] == "mlcommons"
    assert verdict.metadata["tripped"], "no category tripped on plainly hateful content"
    assert set(verdict.metadata["scores"]) <= {"S10", "S11"}

    # A benign reply passes the same rule.
    assert rail.execute("Our refund window is 30 days.").passed is True


def test_a_groundedness_rule_uses_the_run_scoped_context_and_fails_closed_without_it(
    plane: _Rules,
) -> None:
    import fastaiagent as fa
    from fastaiagent.guardrail.from_policy import plane_guardrails_for_agent

    rule = plane.add(
        name="e2e-groundedness",
        implementation_type="groundedness",
        config={"threshold": 0.7, "context_key": "context", "answer_key": "answer"},
        tripwire_message="Answer is not grounded in the retrieved context.",
        action="block",
        severity="high",
    )
    _connect_and_refresh()

    (rail,) = [g for g in plane_guardrails_for_agent(None) if g.name == rule["name"]]
    docs = ["Refunds are issued within 5 business days of approval."]

    with fa.guardrail_context(context=docs):
        supported = rail.execute("Refunds are issued within 5 business days.")
        unsupported = rail.execute("Refunds are instant and we also waive all fees forever.")

    assert supported.passed is True, supported.message
    assert supported.metadata["threshold"] == 0.7
    assert unsupported.passed is False, unsupported.message
    assert unsupported.metadata["unsupported_claims"]

    # With no context the rule cannot run. It must say so rather than scoring the
    # answer against nothing (which would block everything) or itself (nothing).
    blind = rail.execute("Refunds are instant.")
    assert blind.errored is True
    assert blind.passed is False
    assert blind.action_taken == "blocked"
    assert "context" in (blind.message or "")


# --------------------------------------------------------------------------- #
# 5. floor
# --------------------------------------------------------------------------- #
def test_the_organisation_baseline_survives_the_wire(plane: _Rules) -> None:
    from fastaiagent.guardrail.from_policy import plane_guardrails_for_agent

    rule = plane.add(
        name="e2e-floor",
        implementation_type="regex",
        config={"pattern": SSN_PATTERN, "should_match": False},
        tripwire_message="SSN in output.",
        action="warn",
        severity="critical",
        floor=True,
    )
    _connect_and_refresh()

    (rail,) = [g for g in plane_guardrails_for_agent(None) if g.name == rule["name"]]
    assert rail.floor is True
    assert rail.severity == "critical"
    # floor changes no enforcement at the edge — the plane is what stops a
    # project relaxing it. Locally it is context, shown next to the rule.
    assert rail.action == "warn"


# --------------------------------------------------------------------------- #
# 6. The cross-repo contract: what the plane records about what the SDK enforced
#
# Added after the plane fixed the three findings filed in
# claude_files/plane-handover-guardrail-execution-rows.md. Before that fix a
# domain-wide rule — the only kind /policy distributes — produced no
# `guardrail_executions` row at all, so a mask that worked at the edge was
# invisible to the operator who authored it. These are the five assertions the
# plane asked for, and together they are the contract test neither repo can
# write alone.
# --------------------------------------------------------------------------- #
def test_a_domain_wide_rule_records_the_runs_that_did_not_pass(plane: _Rules) -> None:
    """F1. The rule that *acts* is the one whose evidence matters most.

    The original bug wrote a row for a rule that passed and none for the
    domain-wide rule that actually masked — the audit trail recorded the no-ops
    and missed the interventions.
    """
    rule = plane.add(
        name="e2e-rows-domainwide",
        implementation_type="regex",
        config={"pattern": SSN_PATTERN, "should_match": False, "mask_token": "[REDACTED]"},
        tripwire_message="Output contained an SSN.",
        action="mask",
        severity="medium",
    )
    assert rule["project_id"] is None, "this assertion is only meaningful for a domain-wide rule"
    _connect_and_refresh()

    result = _agent(LEAK).run("what is on file?")
    assert "[REDACTED]" in result.output, "the rule did not act, so there is nothing to record"

    rows = _flush_and_await_rows(plane.client, plane.headers, rule["id"], expected=1)
    assert rows, "a domain-wide rule produced no guardrail_executions row"
    row = rows[0]
    assert row["passed"] is False, "the recorded run should be the one that tripped"
    assert row["tripwire_triggered"] is True
    assert row["trigger_point"] == "output"
    assert row["trace_id"], "the row should be joinable back to its trace"


def test_the_row_carries_the_action_the_sdk_took(plane: _Rules) -> None:
    """F2. Step 2 of the original handover's acceptance test, on the SDK path."""
    rule = plane.add(
        name="e2e-rows-detail",
        implementation_type="regex",
        config={"pattern": SSN_PATTERN, "should_match": False, "mask_token": "[REDACTED]"},
        tripwire_message="Output contained an SSN.",
        action="mask",
        severity="high",
    )
    _connect_and_refresh()
    _agent(LEAK).run("what is on file?")

    rows = _flush_and_await_rows(plane.client, plane.headers, rule["id"], expected=1)
    assert rows, "no execution row to inspect"
    detail = rows[0]["result_detail"] or {}
    assert detail.get("action") == "mask", detail
    assert detail.get("action_taken") == "masked", detail
    # The pre-existing key is kept, not replaced — both writers now agree.
    assert "checks" in detail, detail


def test_errored_is_readable_so_a_degraded_control_is_not_read_as_a_clean_block(
    plane: _Rules,
) -> None:
    """F2 follow-on. A block is the control working; an error is it *not* working,
    and under a fail-open policy it means traffic went through unchecked."""
    rule = plane.add(
        name="e2e-rows-errored",
        implementation_type="groundedness",
        # No context will be available, so the check cannot run at all.
        config={"threshold": 0.7},
        tripwire_message="Not grounded.",
        action="block",
        severity="high",
    )
    _connect_and_refresh()

    from fastaiagent._internal.errors import GuardrailBlockedError

    with pytest.raises(GuardrailBlockedError):
        _agent("Refunds are instant.").run("go")

    rows = _flush_and_await_rows(plane.client, plane.headers, rule["id"], expected=1)
    assert rows, "an errored check produced no execution row"
    row = rows[0]
    assert "errored" in row, "GuardrailExecutionRead still omits `errored`"
    assert row["errored"] is True, row
    assert row["passed"] is False


def test_a_rule_of_the_same_name_in_another_domain_never_resolves(plane: _Rules) -> None:
    """F1 tenancy. Widening the resolver from project to domain must not widen it
    past the domain."""
    foreign_domain = _ensure_foreign_domain(plane.client, plane.headers)
    if foreign_domain is None:
        pytest.skip("could not provision a second domain for the tenancy check")

    mine = plane.add(
        name="e2e-rows-tenancy",
        implementation_type="regex",
        config={"pattern": SSN_PATTERN, "should_match": False, "mask_token": "[REDACTED]"},
        tripwire_message="Output contained an SSN.",
        action="mask",
    )
    theirs = plane.add_in_domain(
        foreign_domain,
        name=mine["name"],  # deliberately identical, in a different domain
        implementation_type="regex",
        config={"pattern": SSN_PATTERN, "should_match": False},
        tripwire_message="Someone else's rule.",
        action="block",
    )
    _connect_and_refresh()
    _agent(LEAK).run("what is on file?")

    _flush_and_await_rows(plane.client, plane.headers, mine["id"], expected=1)
    assert _executions(plane.client, plane.headers, mine["id"]), "my own rule recorded nothing"
    assert _executions(plane.client, plane.headers, theirs["id"]) == [], (
        "a same-named rule in another domain absorbed this run's evidence"
    )


def test_deleting_a_rule_with_history_is_refused_not_a_500(plane: _Rules) -> None:
    """F3. Those rows are what the compliance derivation reads, so the plane
    refuses rather than cascading. The refusal has to be legible, though."""
    rule = plane.add(
        name="e2e-rows-delete",
        implementation_type="regex",
        config={"pattern": SSN_PATTERN, "should_match": False},
        tripwire_message="Output contained an SSN.",
        action="block",
    )
    _connect_and_refresh()

    from fastaiagent._internal.errors import GuardrailBlockedError

    with pytest.raises(GuardrailBlockedError):
        _agent(LEAK).run("what is on file?")

    _flush_and_await_rows(plane.client, plane.headers, rule["id"], expected=1)

    resp = plane.client.delete(f"/api/v1/guardrails/{rule['id']}", headers=plane.headers)
    assert resp.status_code == 409, f"expected a refusal, got {resp.status_code}: {resp.text[:200]}"
    assert "deactivate" in resp.text.lower(), "the refusal should name the way out"

    # And the documented way out works.
    assert (
        plane.client.put(
            f"/api/v1/guardrails/{rule['id']}", headers=plane.headers, json={"is_active": False}
        ).status_code
        == 200
    )
