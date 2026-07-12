"""Example 89 — Push an agent that references a governed prompt by slug.

When you build an agent from a control-plane registry prompt, the natural code —
``system_prompt=prompt.format(...)`` — **inlines** the resolved text and loses the
link to the prompt. Pushed that way, the plane stores it as inline and the console
shows the prompt as **"Inline"** with no model/memory linkage.

Set ``Agent(prompt_slug=...)`` instead. ``to_dict()`` then emits the slug and sends
``system_prompt=""``, so the pushed agent **references** the governed prompt (console
shows the slug, memory shows **Enabled**, and the model resolves). Purely additive:
an agent with neither ``prompt_slug`` nor ``memory`` serializes exactly as before.

Usage:
    export FASTAIAGENT_API_KEY=fa_k_...     # key with agent:write + prompt:write
    export FASTAIAGENT_TARGET=http://localhost:20001
    python examples/89_connected_agent_push.py

Expected output (snapshot — real run against a local plane on :20001):
    connected to http://localhost:20001
    published prompt: acme-support-1720800000
    to_dict payload:
      prompt_slug     = acme-support-1720800000
      system_prompt   = '' (empty — the slug is the source of truth)
      memory_enabled  = True
    pushed agent id=<uuid> version=1
    governance: prompt_slug=acme-support-1720800000 memory_enabled=True model=gpt-4o
    done — the console shows the slug (not "Inline") and memory Enabled.
"""

from __future__ import annotations

import os

import httpx

import fastaiagent as fa
from fastaiagent import Agent, AgentMemory, FunctionTool, LLMClient
from fastaiagent._platform.api import get_platform_api
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
    fa.connect(api_key=api_key, target=target)
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

        # 2. Build an agent that REFERENCES the prompt by slug + has memory.
        agent = Agent(
            name="acme-support-bot",
            prompt_slug=slug,
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

        # 4. Push it to the plane.
        resp = get_platform_api().post("/public/v1/sdk/agents", payload)
        agent_id = resp.get("id") or resp.get("agent_id")
        print(f"pushed agent id={agent_id} version={resp.get('version')}")

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
