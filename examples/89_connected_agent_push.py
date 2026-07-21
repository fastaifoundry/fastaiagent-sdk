"""Example 89 — Register an agent as a governed console object (the SDK-owned way).

The SDK owns registration now — no hand-written ``httpx.post(...)``. Three ways in,
all one code path:

* ``fa.connect()`` **auto-registers** (default ON): running or defining-then-
  connecting an agent pushes it. Opt out with ``connect(auto_register=False)``.
* ``agent.push()`` / ``fa.push(agent)`` — explicit, for CI/deploy. Returns a
  ``PushResult`` with ``agent_id``, ``version``, and a clickable console ``url``.
* ``fastaiagent push --module my_app.agents`` — the CLI, at deploy time.

This example uses ``agent.push()`` and also shows governed-input linkage: build the
agent from a control-plane registry **Prompt** (pass it as ``system_prompt``) and
the pushed definition references the slug (console shows the slug, not "Inline"),
with memory **Enabled**. Purely additive — an agent with neither serializes as before.

Usage:
    export FASTAIAGENT_API_KEY=fa_k_...     # key with agent:write + prompt:write
    export FASTAIAGENT_TARGET=http://localhost:20001
    python examples/89_connected_agent_push.py

Expected output (snapshot — real run against a local plane on :20001):
    connected to http://localhost:20001
    published prompt: acme-support-<project>
    to_dict payload:
      prompt_slug     = acme-support-<project>
      system_prompt   = '' (empty — the slug is the source of truth)
      memory_enabled  = True
    pushed agent id=<uuid> version=1 url=http://localhost:20000/next/agents/<uuid>
    governance: prompt_slug=acme-support-<project> memory_enabled=True model=gpt-4o
    done — the console shows the slug (not "Inline") and memory Enabled.
"""

from __future__ import annotations

import os

import httpx

import fastaiagent as fa
from fastaiagent import Agent, AgentMemory, FunctionTool, LLMClient
from fastaiagent.client import _connection
from fastaiagent.prompt import PromptRegistry


def lookup_order(order_id: str) -> str:
    """Look up an order by ID."""
    return f"Order {order_id}: shipped."


def main() -> int:
    api_key = os.environ.get("FASTAIAGENT_API_KEY", "")
    target = os.environ.get("FASTAIAGENT_TARGET", "http://localhost:20001")
    if not api_key:
        print("Skipping: set FASTAIAGENT_API_KEY + FASTAIAGENT_TARGET (agent:write + prompt:write)")
        return 1

    base = target.rstrip("/")
    # auto_register defaults ON — we pass False here so this example pushes
    # explicitly (below) for a deterministic, inspectable result.
    fa.connect(api_key=api_key, target=target, auto_register=False)
    if not _connection.is_connected:
        print("Skipping: connect() did not establish a connection")
        return 1
    print(f"connected to {target}")
    try:
        # 1. Publish a governed prompt (double-brace {{var}} matches the plane).
        registry = PromptRegistry()
        # A stable-ish slug derived from the connected project; no wall clock in
        # the example so it can be re-run (publish always creates a new version).
        slug = f"acme-support-{_connection.project_id or 'demo'}"
        registry.publish(
            slug=slug,
            content=(
                "You are a customer support agent for {{company}}.\n"
                "Be helpful, concise, and professional."
            ),
            variables=["company"],
        )
        print(f"published prompt: {slug}")

        # 2. Build an agent from the registry Prompt (references the slug) + memory.
        #    Passing the Prompt object as system_prompt auto-links prompt_slug and
        #    (on a traced run) stamps the llm span for Prompt Analytics.
        prompt = registry.get(slug, source="platform")
        agent = Agent(
            name="acme-support-bot",
            system_prompt=prompt,
            llm=LLMClient(provider="openai", model="gpt-4o"),
            memory=AgentMemory(),
            tools=[FunctionTool(name="lookup_order", fn=lookup_order)],
        )

        # 3. Inspect the push payload.
        payload = agent.to_dict()
        assert payload["prompt_slug"] == slug
        assert payload["system_prompt"] == ""
        assert payload["memory_enabled"] is True
        print("to_dict payload:")
        print(f"  prompt_slug     = {payload['prompt_slug']}")
        print("  system_prompt   = '' (empty — the slug is the source of truth)")
        print(f"  memory_enabled  = {payload['memory_enabled']}")

        # 4. Register it — the SDK owns the POST. Returns a PushResult with a
        #    clickable console URL. (Equivalent: fa.push(agent).)
        result = agent.push()
        agent_id = result.agent_id
        print(f"pushed agent id={agent_id} version={result.version} url={result.url}")

        # 5. Optional machine check: read governance back.
        if agent_id:
            g = httpx.get(
                f"{base}/api/v1/agents/{agent_id}/governance",
                headers={"X-API-Key": api_key},
            )
            if g.status_code == 200:
                consumes = g.json().get("consumes", {})
                print(
                    f"governance: prompt_slug={consumes.get('prompt_slug')} "
                    f"memory_enabled={consumes.get('memory_enabled')} "
                    f"model={consumes.get('model')}"
                )
            else:
                print(f"governance check skipped: {g.status_code} {g.text[:80]}")

        print('done — the console shows the slug (not "Inline") and memory Enabled.')
        return 0
    finally:
        fa.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
