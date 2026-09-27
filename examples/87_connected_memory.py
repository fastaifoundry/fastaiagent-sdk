"""Example 87 — Connected central memory read (WS3).

When ``connect()``-ed to an Enterprise control plane, an agent can read
**curated, human-approved** durable facts the plane has learned (its central
learning loop extracts facts from already-ingested traces, a human approves them
in the annotation queue, and they're served back). The SDK side is a single
read-only memory block, :class:`PlaneFactBlock`, that injects those facts at the
start of a turn. Per design D5 the SDK has **no fact-push path** — facts are
produced and governed on the plane; the SDK only reads. Recall is **degradable**:
no plane / no facts → the agent still runs.

This demo reads an agent's approved facts back from
``GET /public/v1/memory/facts`` and prints what would be injected. If you provide
a domain-admin login (``E2E_PLANE_EMAIL`` / ``E2E_PLANE_PASSWORD``) it first seeds
one approved fact so the read returns something; otherwise it reads whatever the
plane already has for ``FASTAIAGENT_AGENT_ID``.

See docs/agents/memory.md for the console curation view.

Usage:
    export FASTAIAGENT_API_KEY=fa_k_...          # a connected_state_plane domain key
    export FASTAIAGENT_TARGET=http://localhost:20001
    export FASTAIAGENT_AGENT_ID=<platform agent uuid>      # OR provide admin creds to seed
    export E2E_PLANE_EMAIL=admin@... E2E_PLANE_PASSWORD=...  # optional: seed a fact
    export E2E_PLANE_DOMAIN_ID=<domain uuid>   # which of that admin's domains to seed in
                                               # (default: their first domain)
    python examples/87_connected_memory.py

Expected output (snapshot — real run against a local plane on :20001):
    seeded approved fact for agent ws3-mem-demo      (first run; later runs: "already seeded")
    connected to http://localhost:20001
    reading curated facts for agent=<uuid> ...
    injected 1 fact(s):
      - The customer's preferred contact channel is email.
    agent turn: "How should we contact this customer?"
      -> The customer prefers email.
    done — PlaneFactBlock read governed facts from the plane.

With the plane down, the block degrades and the agent still runs:
    plane unreachable at http://localhost:59999 — running without plane facts.
"""

from __future__ import annotations

import os
import uuid

import fastaiagent as fa
from fastaiagent.agent.memory_blocks import PlaneFactBlock
from fastaiagent.client import _connection

_SEED_FACT = "The customer's preferred contact channel is email."


def _checked(resp, what: str):
    """Fail loudly on a setup call the example can't continue without."""
    if resp.status_code >= 400:
        raise SystemExit(f"{what} failed: HTTP {resp.status_code} {resp.text[:200]}")
    return resp


def _seed_fact_via_admin(base: str) -> str | None:
    """Optional: log in as a domain-admin, push an agent, seed one approved fact.

    Idempotent: the fact is added only if the plane doesn't serve it already.
    Returns the platform agent_id, or None when admin creds aren't provided.
    """
    email = os.environ.get("E2E_PLANE_EMAIL")
    password = os.environ.get("E2E_PLANE_PASSWORD")
    if not (email and password):
        return None

    import httpx

    http = httpx.Client(timeout=30)
    login = _checked(
        http.post(f"{base}/api/v1/auth/login", json={"email": email, "password": password}),
        "admin login",
    )
    jwt = {"Authorization": f"Bearer {login.json()['access_token']}"}
    domains = _checked(http.get(f"{base}/api/v1/users/me/domains", headers=jwt), "domains").json()
    domain_id = os.environ.get("E2E_PLANE_DOMAIN_ID") or domains[0]["id"]
    http.post(
        f"{base}/api/v1/domains/{domain_id}/projects",
        headers=jwt,
        json={"name": "WS3 Demo", "slug": "ws3-demo"},
    )  # 201 or 409 (exists) — both fine

    # Mint a short-lived key just to push the agent definition (no execution).
    kr = _checked(
        http.post(
            f"{base}/api/v1/api-keys",
            headers=jwt,
            json={
                "name": f"ws3-demo-{uuid.uuid4().hex[:6]}",
                "permissions": ["read", "write", "execute"],
                "domain_id": domain_id,
                "scopes": ["agent:write", "agent:execute"],
            },
        ),
        "api key mint",
    )
    key, key_id = kr.json()["key"], kr.json().get("id")
    try:
        ar = _checked(
            http.post(
                f"{base}/public/v1/sdk/agents",
                headers={"X-API-Key": key},
                json={
                    "name": "ws3-mem-demo",
                    "system_prompt": "Support agent.",
                    "agent_type": "single",
                },
            ),
            "agent push",
        )
        agent_id = ar.json().get("id") or ar.json().get("agent_id")
        served = http.get(
            f"{base}/public/v1/memory/facts",
            headers={"X-API-Key": key},
            params={"agent_id": agent_id},
        )
        facts = (
            [f.get("content") for f in served.json().get("facts", [])] if served.is_success else []
        )
        if _SEED_FACT in facts:
            print("already seeded for agent ws3-mem-demo")
        else:
            _checked(
                http.post(
                    f"{base}/api/v1/agents/{agent_id}/memories",
                    headers=jwt,
                    json={"content": _SEED_FACT, "category": "preferences", "importance": 0.9},
                ),
                "fact seed",
            )
            print("seeded approved fact for agent ws3-mem-demo")
    finally:
        if key_id:
            http.delete(f"{base}/api/v1/api-keys/{key_id}", headers=jwt)  # stay under the key cap
    return agent_id


def _reachable(base: str) -> bool:
    import httpx

    try:
        httpx.get(f"{base}/api/v1/health", timeout=3)
        return True
    except httpx.HTTPError:
        return False


def main() -> int:
    api_key = os.environ.get("FASTAIAGENT_API_KEY", "")
    target = os.environ.get("FASTAIAGENT_TARGET", "http://localhost:20001")
    if not api_key:
        print("Skipping: set FASTAIAGENT_API_KEY + FASTAIAGENT_TARGET (connected_state_plane key)")
        return 1

    base = target.rstrip("/")
    plane_up = _reachable(base)
    agent_id = (_seed_fact_via_admin(base) if plane_up else None) or os.environ.get(
        "FASTAIAGENT_AGENT_ID", ""
    )
    if not agent_id:
        print(
            "Skipping: set FASTAIAGENT_AGENT_ID (a platform agent uuid) or admin creds to seed one"
        )
        return 1

    fa.connect(api_key=api_key, target=target)
    print(f"connected to {target}")
    try:
        if not _connection.is_connected:
            print("Skipping: connect() did not establish a connection")
            return 1

        block = PlaneFactBlock(agent_id=agent_id, query_conditioned=False)
        if not plane_up:
            print(f"plane unreachable at {base} — running without plane facts.")
        else:
            print(f"reading curated facts for agent={agent_id} ...")
            msgs = block.render("what does the customer prefer?")
            if not msgs:
                print(
                    "injected 0 facts — none approved for this agent yet (or the key "
                    "lacks access; the block logs a 403/404 once)."
                )
            else:
                # One system message: a bulleted list of approved facts.
                lines = [
                    ln.strip("- ").strip()
                    for ln in msgs[0].content.splitlines()
                    if ln.strip().startswith("-")
                ]
                print(f"injected {len(lines)} fact(s):")
                for ln in lines:
                    print(f"  - {ln}")

        # One real agent turn with the block attached — the plane facts reach the
        # model as a system message. Needs an LLM key; degrades like the block.
        if os.environ.get("OPENAI_API_KEY"):
            agent = fa.Agent(
                name="ws3-mem-demo",
                system_prompt="You are a concise support agent. Use the curated facts.",
                llm=fa.LLMClient(provider="openai", model="gpt-4o-mini"),
                memory=fa.ComposableMemory(blocks=[block]),
            )
            question = "How should we contact this customer?"
            print(f'agent turn: "{question}"')
            print(f"  -> {agent.run(question).output}")
        if plane_up:
            print("done — PlaneFactBlock read governed facts from the plane.")
        else:
            print("done — the agent ran without plane facts (recall is degradable).")
        return 0
    finally:
        fa.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
