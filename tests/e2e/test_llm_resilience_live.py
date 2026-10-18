"""LLMClient fallbacks, pass-through options and current-model images against real providers
(1.91.0, 2E + 2K + X16).

What only a real provider can prove:

* its real context-overflow error is recognised as one, so a fallback with a bigger
  window answers (OpenAI ``context_length_exceeded``, Anthropic "prompt is too long");
* a dead primary falls back to a real model;
* real providers accept the pass-through options the SDK sends;
* the current lineups take images (X16 refused them before any request).

The over-long prompts are rejected before any token is billed; only the fallback's
own call costs (cents).

Run with the keys from ``~/.zshrc``::

    zsh -lc 'unset FASTAIAGENT_API_KEY FASTAIAGENT_TARGET; \
      FASTAIAGENT_LOCAL_DB=<scratch>/local.db \
      .venv/bin/python -m pytest tests/e2e/test_llm_resilience_live.py -m e2e -v -rs'
"""

from __future__ import annotations

import asyncio
import base64
import os
import socket

import pytest

pytestmark = pytest.mark.e2e

PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _need(*keys: str) -> None:
    for key in keys:
        if not os.environ.get(key):
            pytest.skip(f"{key} not set")


def _long_prompt(tokens: int) -> str:
    # ~1 token per short common word.
    return ("the " * tokens) + "\nReply with the single word DONE."


def _ask(client, text: str):
    from fastaiagent.llm.message import UserMessage

    return asyncio.run(client.acomplete([UserMessage(text)]))


def test_openai_context_overflow_falls_back_to_a_bigger_window() -> None:
    _need("OPENAI_API_KEY")
    from fastaiagent import LLMClient

    client = LLMClient(
        provider="openai",
        # The primary needs a small window, so the overflow is cheap to provoke;
        # the fallback is current.
        model="gpt-4o-mini",  # 128k window
        fallbacks=[LLMClient(provider="openai", model="gpt-5-mini")],  # 400k window
    )
    response = _ask(client, _long_prompt(140_000))
    assert response.model.startswith("gpt-5-mini")
    assert "DONE" in (response.content or "").upper()


def test_anthropic_context_overflow_falls_back() -> None:
    _need("ANTHROPIC_API_KEY", "GEMINI_API_KEY")
    from fastaiagent import LLMClient

    client = LLMClient(
        provider="anthropic",
        model="claude-haiku-4-5",  # 200k window
        fallbacks=[LLMClient(provider="gemini", model="gemini-3.5-flash-lite")],
    )
    response = _ask(client, _long_prompt(215_000))
    assert response.content


def test_a_dead_primary_falls_back_to_a_real_model() -> None:
    _need("OPENAI_API_KEY")
    from fastaiagent import LLMClient

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    client = LLMClient(
        provider="openai",
        model="gpt-5-mini",
        api_key="unused",
        base_url=f"http://127.0.0.1:{port}/v1",
        fallbacks=[LLMClient(provider="openai", model="gpt-5-mini")],
    )
    assert _ask(client, "Reply with the single word PONG.").content


def test_real_providers_accept_the_pass_through_options() -> None:
    _need("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    from fastaiagent import LLMClient

    openai = LLMClient(
        provider="openai",
        model="gpt-5-mini",
        extra_body={"safety_identifier": "user-123"},
        extra_headers={"X-Client-Team": "billing"},
    )
    anthropic = LLMClient(
        provider="anthropic",
        model="claude-haiku-4-5",
        extra_body={"metadata": {"user_id": "user-123"}},
        extra_headers={"X-Client-Team": "billing"},
    )
    for client in (openai, anthropic):
        assert _ask(client, "Reply with the single word PONG.").content


@pytest.mark.parametrize(
    ("provider", "model", "key"),
    [
        ("anthropic", "claude-haiku-5-5", "ANTHROPIC_API_KEY"),
        ("openai", "gpt-6-luna", "OPENAI_API_KEY"),
    ],
)
def test_current_models_take_an_image(provider: str, model: str, key: str) -> None:
    _need(key)
    from fastaiagent import LLMClient
    from fastaiagent.llm.message import UserMessage
    from fastaiagent.multimodal.image import Image

    client = LLMClient(provider=provider, model=model)
    response = asyncio.run(
        client.acomplete(
            [
                UserMessage(
                    [
                        "What colour is this pixel? One word.",
                        Image.from_bytes(PIXEL, media_type="image/png"),
                    ]
                )
            ]
        )
    )
    assert response.content


def test_anthropic_without_max_tokens_sends_the_models_maximum() -> None:
    _need("ANTHROPIC_API_KEY")
    from fastaiagent import LLMClient

    client = LLMClient(provider="anthropic", model="claude-haiku-4-5")
    assert client.model_info.max_output_tokens == 64_000
    assert _ask(client, "Reply with the single word OK.").content
