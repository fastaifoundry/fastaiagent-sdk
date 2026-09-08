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
from collections.abc import Iterator
from typing import Any

import pytest

from tests.e2e.conftest import require_env, require_platform

pytestmark = pytest.mark.e2e

SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"
LEAK = "The customer's SSN is 123-45-6789 on file."


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

    def add(self, **body: Any) -> dict[str, Any]:
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

    def cleanup(self) -> None:
        for rule_id in self.created:
            self._c.delete(f"/api/v1/guardrails/{rule_id}", headers=self._h)
        self.created.clear()


@pytest.fixture()
def plane(isolated_local_db: Any) -> Iterator[_Rules]:
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

    plane.add(
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

    mine = next(r for r in rules if r["name"] == "e2e-wire-check")
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

    plane.add(
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
            ("e2e-mask-ui",),
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

    plane.add(
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
    (rail,) = [g for g in plane_guardrails_for_agent(None) if g.name == "e2e-content-safety"]
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

    plane.add(
        name="e2e-groundedness",
        implementation_type="groundedness",
        config={"threshold": 0.7, "context_key": "context", "answer_key": "answer"},
        tripwire_message="Answer is not grounded in the retrieved context.",
        action="block",
        severity="high",
    )
    _connect_and_refresh()

    (rail,) = [g for g in plane_guardrails_for_agent(None) if g.name == "e2e-groundedness"]
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

    plane.add(
        name="e2e-floor",
        implementation_type="regex",
        config={"pattern": SSN_PATTERN, "should_match": False},
        tripwire_message="SSN in output.",
        action="warn",
        severity="critical",
        floor=True,
    )
    _connect_and_refresh()

    (rail,) = [g for g in plane_guardrails_for_agent(None) if g.name == "e2e-floor"]
    assert rail.floor is True
    assert rail.severity == "critical"
    # floor changes no enforcement at the edge — the plane is what stops a
    # project relaxing it. Locally it is context, shown next to the rule.
    assert rail.action == "warn"
