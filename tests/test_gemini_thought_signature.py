"""Gemini thought signatures survive the tool loop (1.91.0, X11) — and results are named (X4).

Every Gemini model after 2.5 — 3.5-flash-lite, 3.6/3.7/3.8-flash, 4-argon and the
``gemini-flash-latest`` / ``gemini-pro-latest`` aliases — attaches an opaque
``thoughtSignature`` to the first ``functionCall`` part of a turn and **rejects the
next request with HTTP 400** when it is not sent back ("Function call is missing a
thought_signature in functionCall parts"). The provider dropped it, so a tool-using
agent on any current Gemini model failed on its second turn, on ``run`` and
``astream`` alike. Measured live before the fix; ``tests/e2e/test_gemini_latest_live.py``
holds the live check.

A history the model did not sign — one that came from another provider, a
``messages=`` replay, or a checkpoint written before 1.91 — carries Google's
validator-skip value on the first call of each unsigned turn, which the API accepts
(also measured live on 2.5, 3.8 and 4-argon).

Real ``Agent`` → real ``LLMClient`` → real ``httpx`` → ``tests.conftest.CaptureServer``
answering exactly as the Gemini API does; the assertions read the request bodies the
SDK actually sent.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer
from fastaiagent.chain.interrupt import Resume, interrupt
from fastaiagent.llm.client import LLMClient
from fastaiagent.llm.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

SIG = "CjQBjz1rX-signature-from-the-model=="
SKIP = "skip_thought_signature_validator"
USAGE = {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}


def get_temperature(city: str) -> str:
    """Temperature in a city."""
    return f"{city}: -4C"


def get_population(city: str) -> str:
    """Population of a city."""
    return f"{city}: 14 million"


TOOLS = [
    FunctionTool(name="get_temperature", fn=get_temperature),
    FunctionTool(name="get_population", fn=get_population),
]

#: Two parallel calls, as Gemini 3 returns them: only the FIRST part is signed.
CALLS = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {"name": "get_temperature", "args": {"city": "Oslo"}},
                        "thoughtSignature": SIG,
                    },
                    {"functionCall": {"name": "get_population", "args": {"city": "Tokyo"}}},
                ],
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": USAGE,
}
ANSWER = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "T=-4C; P=14 million"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": USAGE,
}


def _sse(*bodies: dict[str, Any]) -> str:
    return "".join(f"data: {json.dumps(b)}\n\n" for b in bodies)


def _agent(url: str, **kw: Any) -> Agent:
    return Agent(
        name="gemini-sig",
        llm=LLMClient(provider="gemini", model="gemini-3.8-flash", api_key="k", base_url=url),
        tools=kw.pop("tools", TOOLS),
        **kw,
    )


def _model_turns(request: dict[str, Any]) -> list[list[dict[str, Any]]]:
    return [c["parts"] for c in request["body"]["contents"] if c["role"] == "model"]


def _function_responses(request: dict[str, Any]) -> list[str]:
    return [
        p["functionResponse"]["name"]
        for c in request["body"]["contents"]
        for p in c["parts"]
        if "functionResponse" in p
    ]


def _assert_signed_turn(request: dict[str, Any], signature: str = SIG) -> None:
    (turn,) = _model_turns(request)
    calls = [p for p in turn if "functionCall" in p]
    assert [p["functionCall"]["name"] for p in calls] == ["get_temperature", "get_population"]
    # Sent back exactly where the model put it — the first call — and nowhere else.
    assert calls[0].get("thoughtSignature") == signature
    assert "thoughtSignature" not in calls[1]


# ── X11: the signature goes back ────────────────────────────────────────────


def test_run_sends_the_signature_back(capture_server) -> None:
    capture_server.set_response_sequence([(200, CALLS), (200, ANSWER)])
    result = _agent(capture_server.url).run("Oslo temperature and Tokyo population?")

    assert result.output == "T=-4C; P=14 million"
    _assert_signed_turn(capture_server.requests[1])


def test_astream_sends_the_signature_back(capture_server) -> None:
    capture_server.set_response_sequence(
        [(200, _sse(CALLS), "text/event-stream"), (200, _sse(ANSWER), "text/event-stream")]
    )

    async def drain() -> None:
        async for _ in _agent(capture_server.url).astream("Oslo and Tokyo?"):
            pass

    asyncio.run(drain())
    _assert_signed_turn(capture_server.requests[1])


def test_the_parsed_call_carries_the_signature(capture_server) -> None:
    capture_server.set_response_sequence([(200, CALLS)])
    client = LLMClient(
        provider="gemini", model="gemini-3.8-flash", api_key="k", base_url=capture_server.url
    )
    resp = asyncio.run(client.acomplete([UserMessage("q")]))
    assert [tc.signature for tc in resp.tool_calls] == [SIG, None]


def test_an_unsigned_history_gets_the_validator_skip_value(capture_server) -> None:
    """History the model did not sign (another provider, ``messages=``, an old
    checkpoint) would 400 on Gemini 3; the first call of the unsigned turn carries
    Google's skip value instead."""
    capture_server.set_response_sequence([(200, ANSWER)])
    history = [
        UserMessage("Oslo and Tokyo?"),
        AssistantMessage(
            "",
            tool_calls=[
                ToolCall(id="a1", name="get_temperature", arguments={"city": "Oslo"}),
                ToolCall(id="a2", name="get_population", arguments={"city": "Tokyo"}),
            ],
        ),
        ToolMessage("Oslo: -4C", tool_call_id="a1"),
        ToolMessage("Tokyo: 14 million", tool_call_id="a2"),
    ]
    _agent(capture_server.url).run("Summarise.", messages=history)

    _assert_signed_turn(capture_server.requests[0], signature=SKIP)


def test_a_resumed_run_still_sends_the_signature(capture_server, tmp_path) -> None:
    """The signature is part of the checkpointed history, so the request after a
    resume carries it too."""

    def approve(city: str) -> str:
        """Needs approval."""
        interrupt(reason="approve?", context={"city": city})
        return f"{city}: approved"

    signed_call = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {"name": "approve", "args": {"city": "Oslo"}},
                            "thoughtSignature": SIG,
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": USAGE,
    }
    capture_server.set_response_sequence([(200, signed_call), (200, ANSWER)])
    store = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    agent = _agent(
        capture_server.url, tools=[FunctionTool(name="approve", fn=approve)], checkpointer=store
    )
    paused = agent.run("Approve Oslo.", execution_id="sig-resume")
    assert paused.status == "paused"

    done = agent.resume("sig-resume", resume_value=Resume(approved=True))
    assert done.status == "completed"
    (turn,) = _model_turns(capture_server.requests[-1])
    assert turn[0]["functionCall"]["name"] == "approve"
    assert turn[0]["thoughtSignature"] == SIG


# ── X4: each result is named after its call ────────────────────────────────


def test_function_responses_carry_the_called_name(capture_server) -> None:
    capture_server.set_response_sequence([(200, CALLS), (200, ANSWER)])
    _agent(capture_server.url).run("Oslo temperature and Tokyo population?")
    assert _function_responses(capture_server.requests[1]) == ["get_temperature", "get_population"]


# ── what does NOT change ───────────────────────────────────────────────────


def test_a_non_gemini_checkpoint_has_no_signature_key(tmp_path) -> None:
    """The field is dropped when empty, so checkpoints from every other provider —
    which replicate to the plane — keep their exact shape."""
    from fastaiagent.testing import FunctionModel

    def respond(messages):
        if any(m.role.value == "tool" for m in messages):
            return "done"
        return "", [{"name": "get_temperature", "arguments": {"city": "Oslo"}}]

    store = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    agent = Agent(name="plain", llm=FunctionModel(respond), tools=TOOLS, checkpointer=store)
    agent.run("go", execution_id="plain-run")

    dumped = json.dumps([c.state_snapshot for c in store.list("plain-run")])
    assert "tool_calls" in dumped
    assert '"signature"' not in dumped


@pytest.mark.parametrize("payloads", ["0", "1"])
def test_the_signature_never_reaches_a_span(
    payloads, capture_server, monkeypatch, tmp_path
) -> None:
    """An opaque provider token, not content anyone reads — it stays out of trace
    attributes whatever the payload gate says."""
    from fastaiagent._internal import instance as _instance
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel
    from fastaiagent.trace.otel import get_tracer_provider
    from fastaiagent.trace.storage import TraceStore

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", payloads)
    reset_config()
    _project.set_project_id("test-proj")
    _instance.reset_for_testing()
    otel.reset()
    try:
        capture_server.set_response_sequence([(200, CALLS), (200, ANSWER)])
        result = _agent(capture_server.url).run("Oslo and Tokyo?")
        provider = get_tracer_provider()
        if provider is not None and hasattr(provider, "force_flush"):
            provider.force_flush()
        spans = TraceStore(str(db)).get_trace(result.trace_id).spans
        assert any(s.name.startswith("llm.") for s in spans)
        assert SIG not in json.dumps([s.attributes for s in spans], default=str)
    finally:
        otel.reset()
        _instance.reset_for_testing()
        _project.reset_for_testing()
        reset_config()
