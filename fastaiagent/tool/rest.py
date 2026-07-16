"""RESTTool — calls a REST API endpoint."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from fastaiagent._internal.errors import ToolExecutionError
from fastaiagent.multimodal._http import asafe_http_request
from fastaiagent.tool.base import Tool, ToolResult

logger = logging.getLogger(__name__)

# Defaults match the multimodal fetcher: 30s timeout, 5 redirects, 25 MiB
# body cap. These are tunable per-tool via the constructor.
_DEFAULT_TIMEOUT_SECONDS: float = 30.0
_DEFAULT_MAX_REDIRECTS: int = 5
_DEFAULT_MAX_BYTES: int = 25 * 1024 * 1024


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

    Security:
        SSRF-hardened. The configured ``url`` and any redirect target are
        re-validated against a private-IP block (RFC 1918, loopback,
        link-local incl. cloud-metadata, reserved). When ``body_mapping
        == "path_params"``, the post-substitution URL must keep the same
        host as the template — an LLM-controlled argument cannot pivot
        the request to a different host. Set
        ``FASTAIAGENT_ALLOW_PRIVATE_NETWORKS=1`` to opt in for intranet
        APIs.
    """

    origin = "rest"

    def __init__(
        self,
        name: str,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body_mapping: str = "json_body",
        description: str = "",
        parameters: dict[str, Any] | None = None,
        replay_class: str | None = None,
        *,
        timeout: float | None = None,
        max_retries: int = 0,
        retry_delay: float = 0.5,
        output_type: Any | None = None,
    ):
        self.url = url
        self.method = method.upper()
        self.headers = headers or {}
        self.body_mapping = body_mapping  # json_body, query_params, path_params
        # NOTE: a GET method is NOT auto-classified read_only — the developer
        # must mark replay_class explicitly (replay-safety invariant).
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            replay_class=replay_class,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            output_type=output_type,
        )

    async def aexecute(self, arguments: dict[str, Any], context: Any | None = None) -> ToolResult:
        """Execute the REST API call."""
        try:
            params: dict[str, Any] | None = None
            json_body: Any | None = None
            if self.body_mapping == "query_params":
                params = arguments
            elif self.body_mapping == "json_body":
                json_body = arguments

            url = self.url
            if self.body_mapping == "path_params":
                for key, value in arguments.items():
                    url = url.replace(f"{{{key}}}", str(value))
                # Defence in depth against argument-driven host pivot:
                # the post-substitution URL must keep the template's
                # host. Without this, an LLM could feed a value like
                # ``//attacker/evil`` and rewrite the authority.
                template_host = urlparse(self.url).netloc
                final_host = urlparse(url).netloc
                if template_host and final_host != template_host:
                    raise ToolExecutionError(
                        f"REST tool '{self.name}' refused: path_params "
                        f"substitution changed the host from "
                        f"{template_host!r} to {final_host!r}."
                    )

            resp = await asafe_http_request(
                url,
                method=self.method,
                timeout=_DEFAULT_TIMEOUT_SECONDS,
                max_redirects=_DEFAULT_MAX_REDIRECTS,
                max_bytes=_DEFAULT_MAX_BYTES,
                headers=self.headers,
                json=json_body,
                params=params,
            )

            try:
                output = resp.json()
            except Exception:
                logger.debug("REST tool response is not JSON, using text", exc_info=True)
                output = resp.text

            return ToolResult(
                output=output,
                metadata={"status_code": resp.status_code, "url": str(resp.url)},
            )
        except ToolExecutionError:
            raise
        except Exception as e:
            raise ToolExecutionError(f"REST tool '{self.name}' failed: {e}") from e

    def _tool_type(self) -> str:
        return "rest_api"

    def _config_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "method": self.method,
            "headers": self.headers,
            "body_mapping": self.body_mapping,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> RESTTool:
        config = data.get("config", {})
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parameters=data.get("parameters"),
            replay_class=data.get("replay_class", "side_effecting"),
            url=config.get("url", ""),
            method=config.get("method", "GET"),
            headers=config.get("headers"),
            body_mapping=config.get("body_mapping", "json_body"),
        )
