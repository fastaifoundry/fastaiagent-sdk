"""FastAI client — connect the SDK to the FastAIAgent platform."""

from __future__ import annotations

from typing import Any

from fastaiagent._platform.api import PlatformAPI
from fastaiagent.deploy.push import PushResult, push_all, push_resource


class FastAI:
    """Connect the SDK to the FastAIAgent platform.

    Enables pushing agents, chains, tools, guardrails, and prompts
    to the platform for visual editing, monitoring, and collaboration.

    Example:
        fa = FastAI(api_key="fa_k_...", project="customer-support")

        # Push an agent to the platform
        result = fa.push(my_agent)
        print(f"Pushed: {result.name} (created={result.created})")

        # Push a chain (auto-pushes its agents and tools)
        result = fa.push(my_chain)

        # Batch push
        results = fa.push_all([agent1, agent2, my_chain])
    """

    def __init__(
        self,
        api_key: str,
        target: str = "https://app.fastaiagent.net",
        project: str | None = None,
        timeout: int = 30,
    ):
        self._api = PlatformAPI(
            api_key=api_key,
            base_url=target,
            timeout=timeout,
        )
        self.project = project

    def push(self, resource: Any, **kwargs: Any) -> PushResult:
        """Push an Agent, Chain, Tool, Guardrail, or Prompt to the platform.

        Dependencies are resolved automatically:
        - Pushing a chain auto-pushes its node agents
        - Pushing an agent auto-pushes its tools and guardrails
        """
        return push_resource(self._api, resource, **kwargs)

    def push_all(self, resources: list[Any]) -> list[PushResult]:
        """Push multiple resources in a single batch request."""
        return push_all(self._api, resources)
