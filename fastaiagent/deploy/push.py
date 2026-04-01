"""Push SDK resources to the FastAIAgent platform."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._platform.api import PlatformAPI


class PushResult(BaseModel):
    """Result of pushing a resource to the platform."""

    resource_type: str  # "agent", "chain", "tool", "guardrail", "prompt"
    name: str
    platform_id: str = ""
    created: bool = True  # True if new, False if updated
    dependencies_pushed: list[str] = Field(default_factory=list)


def push_resource(api: PlatformAPI, resource: Any, **kwargs: Any) -> PushResult:
    """Push any SDK resource to the platform.

    Dispatches based on resource type. Dependencies (tools, guardrails)
    are automatically included in the push.
    """
    from fastaiagent.agent.agent import Agent
    from fastaiagent.chain.chain import Chain
    from fastaiagent.guardrail.guardrail import Guardrail
    from fastaiagent.prompt.prompt import Prompt
    from fastaiagent.tool.base import Tool

    if isinstance(resource, Agent):
        return _push_agent(api, resource)
    elif isinstance(resource, Chain):
        return _push_chain(api, resource)
    elif isinstance(resource, Tool):
        return _push_tool(api, resource)
    elif isinstance(resource, Guardrail):
        return _push_guardrail(api, resource)
    elif isinstance(resource, Prompt):
        return _push_prompt(api, resource)
    else:
        raise TypeError(
            f"Cannot push {type(resource).__name__}. "
            f"Pushable types: Agent, Chain, Tool, Guardrail, Prompt."
        )


def push_all(api: PlatformAPI, resources: list[Any]) -> list[PushResult]:
    """Batch push multiple resources via /sdk/push endpoint.

    Collects all resources and their dependencies into a single request.
    """
    from fastaiagent.agent.agent import Agent
    from fastaiagent.chain.chain import Chain
    from fastaiagent.guardrail.guardrail import Guardrail
    from fastaiagent.prompt.prompt import Prompt
    from fastaiagent.tool.base import Tool

    payload: dict[str, list[dict]] = {
        "tools": [],
        "guardrails": [],
        "agents": [],
        "chains": [],
        "prompts": [],
    }

    for r in resources:
        if isinstance(r, Agent):
            payload["agents"].append(r.to_dict())
        elif isinstance(r, Chain):
            payload["chains"].append(r.to_dict())
        elif isinstance(r, Tool):
            payload["tools"].append(r.to_dict())
        elif isinstance(r, Guardrail):
            payload["guardrails"].append(r.to_dict())
        elif isinstance(r, Prompt):
            payload["prompts"].append(r.to_dict())

    result = api.post("/public/v1/sdk/push", payload)

    results = []
    for item in result.get("created", []):
        rtype, name = item.split(":", 1)
        results.append(PushResult(resource_type=rtype, name=name, created=True))
    for item in result.get("updated", []):
        rtype, name = item.split(":", 1)
        results.append(PushResult(resource_type=rtype, name=name, created=False))
    return results


def _push_agent(api: PlatformAPI, agent: Any) -> PushResult:
    """Push an agent with its tools and guardrails."""
    data = agent.to_dict()
    result = api.post("/public/v1/sdk/push", {
        "agents": [data],
    })
    created_items = result.get("created", [])
    updated_items = result.get("updated", [])

    deps = [i for i in created_items + updated_items if not i.startswith("agent:")]
    was_created = any(f"agent:{agent.name}" in i for i in created_items)

    return PushResult(
        resource_type="agent",
        name=agent.name,
        platform_id=result.get("id", ""),
        created=was_created,
        dependencies_pushed=deps,
    )


def _push_chain(api: PlatformAPI, chain: Any) -> PushResult:
    """Push a chain. Node agents are pushed separately if attached."""
    payload: dict[str, list[dict]] = {"chains": [chain.to_dict()], "agents": []}

    # Auto-push agents attached to nodes
    for node in chain.nodes:
        if node.agent is not None:
            payload["agents"].append(node.agent.to_dict())

    result = api.post("/public/v1/sdk/push", payload)
    created_items = result.get("created", [])
    updated_items = result.get("updated", [])

    deps = [i for i in created_items + updated_items if not i.startswith("chain:")]
    was_created = any(f"chain:{chain.name}" in i for i in created_items)

    return PushResult(
        resource_type="chain",
        name=chain.name,
        created=was_created,
        dependencies_pushed=deps,
    )


def _push_tool(api: PlatformAPI, tool: Any) -> PushResult:
    result = api.post("/public/v1/sdk/tools", tool.to_dict())
    return PushResult(
        resource_type="tool",
        name=tool.name,
        platform_id=result.get("id", ""),
        created=result.get("created", True),
    )


def _push_guardrail(api: PlatformAPI, guardrail: Any) -> PushResult:
    result = api.post("/public/v1/sdk/guardrails", guardrail.to_dict())
    return PushResult(
        resource_type="guardrail",
        name=guardrail.name,
        platform_id=result.get("id", ""),
        created=result.get("created", True),
    )


def _push_prompt(api: PlatformAPI, prompt: Any) -> PushResult:
    result = api.post("/public/v1/sdk/prompts", prompt.to_dict())
    return PushResult(
        resource_type="prompt",
        name=prompt.name,
        platform_id=result.get("id", ""),
        created=result.get("created", True),
    )
