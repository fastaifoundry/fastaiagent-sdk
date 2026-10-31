"""A 200 whose body is not JSON raises LLMProviderError (1.95.0, S8).

A proxy or a wrong ``base_url`` answers 200 with an HTML page. The SDK raised a
bare ``json.JSONDecodeError``, which escaped the ``except LLMError`` that the
error-handling docs tell users to write. A real localhost server serves the page.
"""

from __future__ import annotations

from typing import Any

import pytest

from fastaiagent import LLMClient
from fastaiagent._internal.errors import LLMError, LLMProviderError
from fastaiagent.llm.message import UserMessage

HTML = "<html><body>Sign in to the corporate proxy</body></html>"


@pytest.mark.parametrize(
    ("provider", "api"),
    [
        ("custom", "chat"),
        ("openai", "responses"),
        ("anthropic", "chat"),
        ("ollama", "chat"),
        ("gemini", "chat"),
    ],
)
def test_an_html_200_is_a_provider_error(capture_server: Any, provider: str, api: str) -> None:
    capture_server.set_response_sequence([(200, HTML, "text/html")])
    kwargs: dict[str, Any] = {"api": api} if api != "chat" else {}
    client = LLMClient(
        provider=provider, model="m", api_key="k", base_url=capture_server.url, **kwargs
    )
    with pytest.raises(LLMProviderError, match="not JSON") as raised:
        client.complete([UserMessage("hi")])
    assert isinstance(raised.value, LLMError)
    assert raised.value.status_code == 200
    assert "corporate proxy" in str(raised.value)


@pytest.mark.parametrize(
    ("provider", "api", "line"),
    [
        ("custom", "chat", "data: {not json"),
        ("openai", "responses", "data: {not json"),
        ("anthropic", "chat", "data: {not json"),
        ("ollama", "chat", "{not json"),
    ],
)
def test_a_streamed_chunk_that_is_not_json_is_a_provider_error(
    capture_server: Any, provider: str, api: str, line: str
) -> None:
    capture_server.set_response_sequence([(200, f"{line}\n\n", "text/event-stream")])
    kwargs: dict[str, Any] = {"api": api} if api != "chat" else {}
    client = LLMClient(
        provider=provider, model="m", api_key="k", base_url=capture_server.url, **kwargs
    )
    with pytest.raises(LLMProviderError, match="not JSON"):
        list(client.stream([UserMessage("hi")]))
