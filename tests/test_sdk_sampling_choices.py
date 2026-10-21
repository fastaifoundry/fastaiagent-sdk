"""What the SDK sends for itself, on models that refuse it (1.91.0, X22–X24).

Measured against the live APIs on 2026-10-19: ``temperature=0`` is a 400 on
gpt-5-mini (any reasoning effort), gpt-5.5, gpt-5.6, the GPT-6 family and Claude
5.x ("`temperature` is deprecated for this model"), and on gpt-5.1 / 5.2 / 5.4
once reasoning is on. gpt-4o-mini, gpt-5.1 at its default effort and
claude-haiku-4-5 accept it.

* X23 — the ``topic``, ``content_safety`` and ``groundedness`` judges always sent
  ``temperature=0``, so on those models every check errored and ``on_error``
  blocked every request. They now send it only where the model takes it.
* X24 — ``with_determinism("deterministic")`` forced ``temperature=0`` on the
  rebuilt client: every deterministic rerun of a GPT-5 or Claude 5 trace was a
  400. It now pins ``seed=42`` always and the temperature only where accepted.
* X22 — a rule's ``llm`` config refused ``reasoning`` / ``cache`` / ``api``, the
  keys ``LLMClient.to_dict()`` gained in 1.91.0, so a judge could not be told
  to reason less, and a ``to_dict()`` of such a client was refused as a config.

Real ``LLMClient`` → real ``httpx`` → ``tests.conftest.CaptureServer``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailType
from fastaiagent.llm.client import LLMClient
from fastaiagent.llm.param_rules import pins_temperature

OPENAI_REPLY = {
    "id": "c",
    "object": "chat.completion",
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "{}"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


def _anthropic_reply(text: str) -> dict[str, Any]:
    return {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "model": "x",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }


@pytest.mark.parametrize(
    ("provider", "model", "reasoning", "pins"),
    [
        ("openai", "gpt-4o-mini", None, True),
        ("openai", "gpt-5.1", None, True),
        ("openai", "gpt-5.1", "none", True),
        ("openai", "gpt-5.1", "low", False),
        ("openai", "gpt-5.4", "medium", False),
        ("openai", "gpt-5-mini", None, False),
        ("openai", "gpt-5-mini", "minimal", False),
        ("openai", "gpt-5.5", None, False),
        ("openai", "gpt-6-luna", None, False),
        ("openai", "gpt-6.1-sol", None, False),
        ("anthropic", "claude-haiku-4-5", None, True),
        ("anthropic", "claude-haiku-4-5", "low", False),
        ("anthropic", "claude-sonnet-5-5", None, False),
        ("gemini", "gemini-3.8-flash", None, True),
    ],
)
def test_which_models_take_temperature_zero(provider, model, reasoning, pins) -> None:
    assert pins_temperature(provider, model, reasoning) is pins


# ── X23: the judges ─────────────────────────────────────────────────────────


def _rule(kind: str, llm: dict[str, Any]) -> Guardrail:
    configs = {
        "topic": {"topics": ["violence"], "mode": "deny"},
        "content_safety": {"categories": ["S1"]},
        "groundedness": {"threshold": 0.5},
    }
    return Guardrail(
        name=f"judge_{kind}",
        guardrail_type=GuardrailType(kind),
        position=GuardrailPosition.output,
        config={**configs[kind], "llm": llm},
    )


@pytest.mark.parametrize("kind", ["topic", "content_safety", "groundedness"])
@pytest.mark.parametrize(
    ("model", "extra", "sends_zero"),
    [
        ("gpt-4o-mini", {}, True),
        ("gpt-5-mini", {}, False),
        ("gpt-5.1", {"reasoning": "low"}, False),
        ("gpt-6-luna", {}, False),
    ],
)
def test_openai_judges_send_temperature_only_where_accepted(
    capture_server, monkeypatch, kind: str, model: str, extra: dict, sends_zero: bool
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    capture_server.set_response_sequence([(200, OPENAI_REPLY)])
    rule = _rule(
        kind, {"provider": "openai", "model": model, "base_url": capture_server.url, **extra}
    )
    # groundedness reads its context and answer from the payload (the plane's shape).
    payload = (
        {"context": "The sky is blue.", "answer": "The sky is blue."}
        if kind == "groundedness"
        else "The sky is blue."
    )
    rule.execute(payload)
    (req,) = capture_server.requests
    if sends_zero:
        assert req["body"]["temperature"] == 0
    else:
        assert "temperature" not in req["body"]


@pytest.mark.parametrize(
    ("model", "sends_zero"), [("claude-haiku-4-5", True), ("claude-sonnet-5-5", False)]
)
def test_anthropic_judges_send_temperature_only_where_accepted(
    capture_server, monkeypatch, model: str, sends_zero: bool
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    capture_server.set_response_sequence([(200, _anthropic_reply("none"))])
    rule = _rule("topic", {"provider": "anthropic", "model": model, "base_url": capture_server.url})
    rule.execute("hello")
    body = capture_server.requests[0]["body"]
    assert ("temperature" in body) is sends_zero


# ── X22: the judge's llm config ─────────────────────────────────────────────


def test_a_rule_llm_config_takes_every_key_llmclient_serializes() -> None:
    client = LLMClient(
        provider="openai", model="gpt-5.1", reasoning="low", cache=True, api="responses"
    )
    config = client.to_dict()
    assert {"reasoning", "cache", "api"} <= set(config)
    rule = _rule("topic", config)  # refused before 1.91.0's fix
    assert rule.config["llm"]["reasoning"] == "low"
    # Still never a secret.
    with pytest.raises(ValueError, match="api_key"):
        _rule("topic", {**config, "api_key": "sk-x"})


# ── X24: deterministic replay ───────────────────────────────────────────────


def _trace(model: str, base_url: str) -> Any:
    from fastaiagent.trace.storage import SpanData, TraceData

    root = SpanData(
        span_id="root",
        trace_id="t-det",
        name="agent.bot",
        start_time="2026-10-19T00:00:00Z",
        end_time="2026-10-19T00:00:01Z",
        attributes={
            "agent.name": "bot",
            "agent.input": "hi",
            "agent.output": "hello",
            "agent.config": json.dumps({}),
            "agent.tools": json.dumps([]),
            "agent.guardrails": json.dumps([]),
            "agent.llm.provider": "openai",
            "agent.llm.model": model,
            "agent.llm.config": json.dumps(
                {"provider": "openai", "model": model, "base_url": base_url}
            ),
        },
    )
    return TraceData(trace_id="t-det", name="agent.bot", start_time="", end_time="", spans=[root])


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "sends_zero"), [("gpt-4o-mini", True), ("gpt-5-mini", False)])
async def test_deterministic_replay_pins_what_the_model_takes(
    capture_server, monkeypatch, model: str, sends_zero: bool
) -> None:
    from fastaiagent.trace.replay import Replay

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    capture_server.set_response_sequence(
        [
            (
                200,
                {
                    **OPENAI_REPLY,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        ]
    )
    forked = Replay(_trace(model, capture_server.url)).fork_at(step=0)
    result = await forked.with_determinism("deterministic").arerun()
    assert result.new_output == "hello"
    body = capture_server.requests[0]["body"]
    assert body["seed"] == 42
    if sends_zero:
        assert body["temperature"] == 0
    else:
        assert "temperature" not in body
