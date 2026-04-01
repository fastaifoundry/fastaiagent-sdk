"""RESTTool — calls a REST API endpoint."""

from __future__ import annotations

from typing import Any

from fastaiagent._internal.errors import ToolExecutionError
from fastaiagent.tool.base import Tool, ToolResult


class RESTTool(Tool):
    """A tool that calls a REST API endpoint.

    Example:
        tool = RESTTool(
            name="weather",
            description="Get weather for a city",
            url="https://api.weather.com/v1/forecast",
            method="GET",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
    """

    def __init__(
        self,
        name: str,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body_mapping: str = "json_body",
        description: str = "",
        parameters: dict | None = None,
    ):
        self.url = url
        self.method = method.upper()
        self.headers = headers or {}
        self.body_mapping = body_mapping  # json_body, query_params, path_params
        super().__init__(name=name, description=description, parameters=parameters)

    async def aexecute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the REST API call."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                kwargs: dict[str, Any] = {"headers": self.headers}

                if self.body_mapping == "query_params":
                    kwargs["params"] = arguments
                elif self.body_mapping == "json_body":
                    kwargs["json"] = arguments

                url = self.url
                if self.body_mapping == "path_params":
                    for key, value in arguments.items():
                        url = url.replace(f"{{{key}}}", str(value))

                resp = await client.request(self.method, url, **kwargs)
                resp.raise_for_status()

                try:
                    output = resp.json()
                except Exception:
                    output = resp.text

                return ToolResult(
                    output=output,
                    metadata={"status_code": resp.status_code, "url": str(resp.url)},
                )
        except Exception as e:
            raise ToolExecutionError(f"REST tool '{self.name}' failed: {e}") from e

    def _tool_type(self) -> str:
        return "rest_api"

    def _config_dict(self) -> dict:
        return {
            "url": self.url,
            "method": self.method,
            "headers": self.headers,
            "body_mapping": self.body_mapping,
        }

    @classmethod
    def _from_dict(cls, data: dict) -> RESTTool:
        config = data.get("config", {})
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parameters=data.get("parameters"),
            url=config.get("url", ""),
            method=config.get("method", "GET"),
            headers=config.get("headers"),
            body_mapping=config.get("body_mapping", "json_body"),
        )
