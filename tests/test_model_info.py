"""Per-model data: ``LLMClient.model_info`` (1.91.0, 2F) and the vision / PDF registry (X16).

``model_info`` answers what a model can take and give — context window, max output
tokens, whether it reasons, caches prompts, reads images and PDFs — from one
prefix-matched table next to pricing. Every number is from the provider's own
source, read 2026-10-18 (claude_files/model_data_2026-10-18.md): OpenAI's model
pages, Anthropic's Models API (``max_input_tokens`` / ``max_tokens``) and Gemini's
Models API (``inputTokenLimit`` / ``outputTokenLimit``). An unknown model gets
``None``, never a guess. Vision and PDF come from the existing multimodal registry
rather than a second copy.

Anthropic and Bedrock no longer cap an unset ``max_tokens`` at 4096: they send the
model's own maximum (checked live — the API accepts it on non-streamed calls).

X16: the vision / PDF registry predated Claude 5.x and GPT-6, so an image sent to
``claude-haiku-5-5`` or ``gpt-6-luna`` raised ``NonVisionModelError`` before any
request was made (measured live), and a PDF to Claude 5.x was reduced to text.
"""

from __future__ import annotations

import asyncio

import pytest

from fastaiagent.llm.client import LLMClient
from fastaiagent.llm.message import UserMessage
from fastaiagent.multimodal.registry import is_vision_capable, supports_native_pdf


def _info(provider: str, model: str):
    return LLMClient(provider=provider, model=model, api_key="k").model_info


@pytest.mark.parametrize(
    ("provider", "model", "context", "max_out", "reasoning"),
    [
        ("openai", "gpt-6-astra", 1_050_000, 128_000, True),
        ("openai", "gpt-6.1-sol", 1_050_000, 128_000, True),
        ("openai", "gpt-6-luna", 1_050_000, 128_000, True),
        ("openai", "gpt-5.1", 400_000, 128_000, True),
        ("openai", "gpt-5-mini", 400_000, 128_000, True),
        ("openai", "gpt-5-nano", 400_000, 128_000, True),
        ("openai", "gpt-4.1", 1_047_576, 32_768, False),
        ("openai", "gpt-4.1-mini", 1_047_576, 32_768, False),
        ("openai", "gpt-4o", 128_000, 16_384, False),
        ("openai", "gpt-4o-mini", 128_000, 16_384, False),
        ("openai", "o3", 200_000, 100_000, True),
        ("openai", "o4-mini", 200_000, 100_000, True),
        ("anthropic", "claude-opus-5-5", 1_000_000, 128_000, True),
        ("anthropic", "claude-haiku-5-5", 1_000_000, 128_000, True),
        ("anthropic", "claude-fable-5-1", 1_000_000, 128_000, True),
        ("anthropic", "claude-sonnet-4-6", 1_000_000, 128_000, True),
        ("anthropic", "claude-haiku-4-5", 200_000, 64_000, True),
        ("anthropic", "claude-haiku-4-5-20251001", 200_000, 64_000, True),
        ("anthropic", "claude-sonnet-4-5", 200_000, 64_000, True),
        ("gemini", "gemini-4-argon", 1_048_576, 32_768, True),
        ("gemini", "gemini-3.8-flash", 1_048_576, 65_536, True),
        ("gemini", "gemini-3.5-flash-lite", 1_048_576, 65_536, True),
        ("gemini", "gemini-2.5-flash", 1_048_576, 65_536, True),
        ("gemini", "gemini-flash-latest", 1_048_576, 65_536, True),
        ("bedrock", "anthropic.claude-haiku-4-5-20251001-v1:0", 200_000, 64_000, True),
        ("bedrock", "us.anthropic.claude-opus-5-5", 1_000_000, 128_000, True),
    ],
)
def test_known_models(provider, model, context, max_out, reasoning) -> None:
    info = _info(provider, model)
    assert (info.context_window, info.max_output_tokens) == (context, max_out)
    assert info.supports_reasoning is reasoning
    assert info.supports_cache is True


def test_longest_prefix_wins() -> None:
    assert _info("openai", "gpt-4.1-mini").max_output_tokens == 32_768
    assert _info("openai", "gpt-5-mini-2025-08-07").context_window == 400_000


def test_an_unknown_model_is_unknown_not_guessed() -> None:
    info = _info("openai", "my-private-fine-tune")
    assert info.context_window is None and info.max_output_tokens is None
    assert info.supports_reasoning is None and info.supports_cache is None


def test_vision_and_pdf_come_from_the_multimodal_registry() -> None:
    info = _info("anthropic", "claude-sonnet-5-5")
    assert info.supports_vision is True and info.supports_pdf is True
    gpt6 = _info("openai", "gpt-6-luna")
    assert gpt6.supports_vision is True and gpt6.supports_pdf is False  # no file input
    assert _info("openai", "gpt-3.5-turbo").supports_vision is False


# ── X16: the registry knows the current lineups ────────────────────────────


@pytest.mark.parametrize(
    ("provider", "model", "pdf"),
    [
        ("anthropic", "claude-opus-5-5", True),
        ("anthropic", "claude-sonnet-5-5", True),
        ("anthropic", "claude-haiku-5-5", True),
        ("anthropic", "claude-fable-5-1", True),
        ("anthropic", "claude-sonnet-5", True),
        ("bedrock", "anthropic.claude-opus-5-5", True),
        ("bedrock", "us.anthropic.claude-haiku-5-5", True),
        ("openai", "gpt-6-astra", False),
        ("openai", "gpt-6.1-sol", False),
        ("openai", "gpt-6-luna", False),
    ],
)
def test_current_models_take_images(provider: str, model: str, pdf: bool) -> None:
    assert is_vision_capable(provider, model)
    assert supports_native_pdf(provider, model) is pdf


def test_an_image_to_claude_5_is_sent_not_refused(capture_server) -> None:
    import base64

    from fastaiagent.multimodal.image import Image

    capture_server.set_response_sequence(
        [
            (
                200,
                {
                    "id": "m1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-5-5",
                    "content": [{"type": "text", "text": "red"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 9, "output_tokens": 1},
                },
            )
        ]
    )
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    client = LLMClient(
        provider="anthropic", model="claude-haiku-5-5", api_key="k", base_url=capture_server.url
    )
    asyncio.run(
        client.acomplete([UserMessage(["colour?", Image.from_bytes(png, media_type="image/png")])])
    )
    content = capture_server.requests[0]["body"]["messages"][0]["content"]
    assert any(block.get("type") == "image" for block in content)


# ── Anthropic / Bedrock default max_tokens ─────────────────────────────────


ANTHROPIC_OK = {
    "id": "m1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3, "output_tokens": 1},
}


@pytest.mark.parametrize(
    ("model", "sent"),
    [("claude-haiku-4-5", 64_000), ("claude-opus-5-5", 128_000), ("claude-private-x", 4096)],
)
def test_anthropic_sends_the_models_own_max_output(model: str, sent: int, capture_server) -> None:
    capture_server.set_response_sequence([(200, ANTHROPIC_OK)])
    client = LLMClient(provider="anthropic", model=model, api_key="k", base_url=capture_server.url)
    asyncio.run(client.acomplete([UserMessage("q")]))
    assert capture_server.requests[0]["body"]["max_tokens"] == sent


def test_an_explicit_max_tokens_still_wins(capture_server) -> None:
    capture_server.set_response_sequence([(200, ANTHROPIC_OK), (200, ANTHROPIC_OK)])
    client = LLMClient(
        provider="anthropic",
        model="claude-haiku-4-5",
        api_key="k",
        base_url=capture_server.url,
        max_tokens=500,
    )
    asyncio.run(client.acomplete([UserMessage("q")]))
    asyncio.run(client.acomplete([UserMessage("q")], max_tokens=77))
    assert [r["body"]["max_tokens"] for r in capture_server.requests] == [500, 77]


def test_model_info_is_exported() -> None:
    from fastaiagent import ModelInfo
    from fastaiagent.llm import ModelInfo as FromLLM

    assert ModelInfo is FromLLM
