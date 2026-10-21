"""Tools the model's provider runs (1.91.0, 2D): ``web_search()``,
``code_execution()``, ``file_search()``.

A ``ProviderTool`` is sent in each provider's own shape and never run here. What
these tests pin, each measured against the live APIs on 2026-10-19:

* **The turn goes back verbatim.** Anthropic runs a search requested beside a
  client tool only once the turn is sent back with its ``server_tool_use`` block —
  stripped, the model searches again and both are billed. Gemini 3+ 400s when its
  ``toolCall`` / ``toolResponse`` parts are missing. The turn rides on
  ``Message.provider_content`` through the loop, ``astream``, checkpoints and resume.
* **Each call is recorded** as a ``tool.<name>`` span (``tool.origin="provider"``)
  and a tool-call record, and judged by the ``tool_result`` guardrails.
* **A ``tool_call`` guardrail cannot check it** (the arguments exist only inside the
  model call): could-not-run, ``on_error`` decides — ``"block"`` before the model
  is called, ``"allow"`` recorded unchecked per call (decisions B2-6, b2-d17).
* **Per-call fees** are priced with the tokens (b2-d16).
* **A provider that runs none of them refuses** — never a silently dropped tool.

Real ``LLMClient`` → real ``httpx`` → ``tests.conftest.CaptureServer`` answering
in each provider's shape.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from fastaiagent import Agent, Chain, FunctionTool, SQLiteCheckpointer
from fastaiagent._internal.errors import GuardrailBlockedError, LLMError
from fastaiagent._internal.pricing import (
    run_cost,
    set_tool_fee_overrides,
    start_run_cost,
    stop_run_cost,
    tool_fees_cost,
)
from fastaiagent.agent.executor import _deserialize_messages, _serialize_messages
from fastaiagent.chain.interrupt import Resume, interrupt
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailResult, GuardrailType
from fastaiagent.llm.client import LLMClient
from fastaiagent.llm.message import AssistantMessage, ProviderContent, UserMessage
from fastaiagent.llm.stream import ProviderTurn, TextDelta
from fastaiagent.tool import ProviderTool, Tool, code_execution, file_search, web_search


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"{city}: 14C, rain"


WEATHER = FunctionTool(name="get_weather", fn=get_weather)


# ── Anthropic shapes ────────────────────────────────────────────────────────

SEARCH_USE = {
    "type": "server_tool_use",
    "id": "srvtoolu_1",
    "name": "web_search",
    "input": {"query": "latest stable Python release"},
}
WEATHER_USE = {
    "type": "tool_use",
    "id": "toolu_1",
    "name": "get_weather",
    "input": {"city": "Paris"},
}
SEARCH_RESULT = {
    "type": "web_search_tool_result",
    "tool_use_id": "srvtoolu_1",
    "content": [
        {
            "type": "web_search_result",
            "title": "Python Release Python 3.15.0",
            "url": "https://www.python.org/downloads/release/python-3150/",
            "encrypted_content": "Eq-opaque==",
            "page_age": "October 9, 2026",
        }
    ],
}
CITED_TEXT = {
    "type": "text",
    "text": "Python 3.15 is the latest stable release.",
    "citations": [
        {
            "type": "web_search_result_location",
            "url": "https://www.python.org/downloads/release/python-3150/",
            "title": "Python Release Python 3.15.0",
            "encrypted_index": "x",
            "cited_text": "Python 3.15.0 is the newest major release",
        }
    ],
}


def _anthropic(content: list[dict[str, Any]], stop: str = "end_turn", searches: int = 0) -> dict:
    usage: dict[str, Any] = {"input_tokens": 1000, "output_tokens": 100}
    if searches:
        usage["server_tool_use"] = {"web_search_requests": searches, "web_fetch_requests": 0}
    return {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": content,
        "stop_reason": stop,
        "usage": usage,
    }


def _claude(url: str) -> LLMClient:
    return LLMClient(provider="anthropic", model="claude-sonnet-5-5", api_key="k", base_url=url)


def _search_then_answer(capture_server: Any) -> None:
    """Turn 1: the search is requested beside a client tool and does NOT run (no
    result, no search counted) — the live behaviour. Turn 2: it ran, first."""
    capture_server.set_response_sequence(
        [
            (200, _anthropic([SEARCH_USE, WEATHER_USE], stop="tool_use")),
            (200, _anthropic([SEARCH_RESULT, CITED_TEXT], searches=1)),
        ]
    )


# ── the tool itself ─────────────────────────────────────────────────────────


def test_factories_round_trip_and_never_run_locally() -> None:
    tool = web_search(max_uses=2, allowed_domains=["python.org"])
    assert isinstance(tool, ProviderTool) and tool.origin == "provider"
    data = tool.to_dict()
    assert data["tool_type"] == "provider"
    assert data["config"] == {
        "kind": "web_search",
        "options": {"max_uses": 2, "allowed_domains": ["python.org"]},
    }
    again = Tool.from_dict(data)
    assert isinstance(again, ProviderTool) and again.options == tool.options
    result = code_execution().execute({})
    assert result.error and "provider" in result.error
    with pytest.raises(ValueError, match="at least one store"):
        file_search([])
    agent = Agent(name="a", llm=LLMClient(provider="anthropic"), tools=[web_search(), WEATHER])
    rebuilt = Agent.from_dict(agent.to_dict())
    assert [type(t).__name__ for t in rebuilt.tools] == ["ProviderTool", "FunctionTool"]


def test_an_mcp_server_does_not_expose_them() -> None:
    pytest.importorskip("mcp")
    from fastaiagent.tool.mcp_server import FastAIAgentMCPServer

    agent = Agent(name="a", llm=LLMClient(provider="anthropic"), tools=[web_search(), WEATHER])
    server = FastAIAgentMCPServer(agent, expose_tools=True)
    assert [t.name for t in server._inner_tools] == ["get_weather"]


@pytest.mark.parametrize(
    ("client_kw", "where"),
    [
        ({"provider": "openai", "model": "gpt-5.1"}, "Chat Completions"),
        ({"provider": "ollama", "model": "llama3"}, "Ollama"),
        ({"provider": "groq", "model": "llama-3.3-70b"}, "groq"),
    ],
)
def test_a_provider_that_runs_none_refuses_before_sending(
    capture_server, client_kw: dict[str, Any], where: str
) -> None:
    llm = LLMClient(api_key="k", base_url=capture_server.url, **client_kw)
    agent = Agent(name="a", llm=llm, tools=[web_search()])
    with pytest.raises(LLMError, match=where):
        agent.run("hi")
    assert capture_server.requests == []


def test_bedrock_refuses_too() -> None:
    pytest.importorskip("boto3")
    llm = LLMClient(provider="bedrock", model="anthropic.claude-sonnet-5-5", region="us-east-1")
    with pytest.raises(LLMError, match="Bedrock"):
        llm._bedrock_converse_body([UserMessage("hi")], [web_search().to_openai_format()], {})


@pytest.mark.parametrize(
    ("client_kw", "tool", "match"),
    [
        ({"provider": "anthropic"}, file_search(["vs_1"]), "no hosted file_search"),
        ({"provider": "gemini"}, web_search(allowed_domains=["a.org"]), "allowed_domains"),
        ({"provider": "openai", "api": "responses"}, web_search(max_uses=1), "max_uses"),
        (
            {"provider": "anthropic"},
            web_search(allowed_domains=["a"], blocked_domains=["b"]),
            "not both",
        ),
    ],
)
def test_an_option_a_provider_cannot_honour_is_refused(
    capture_server, client_kw: dict[str, Any], tool: ProviderTool, match: str
) -> None:
    llm = LLMClient(api_key="k", base_url=capture_server.url, **client_kw)
    with pytest.raises(LLMError, match=match):
        Agent(name="a", llm=llm, tools=[tool]).run("hi")
    assert capture_server.requests == []


# ── Anthropic ───────────────────────────────────────────────────────────────


def test_anthropic_server_tools_are_sent_in_its_shape(capture_server) -> None:
    capture_server.set_response_sequence([(200, _anthropic([{"type": "text", "text": "ok"}]))])
    agent = Agent(
        name="a",
        llm=_claude(capture_server.url),
        tools=[
            web_search(max_uses=2, blocked_domains=["x.com"], user_location={"city": "Paris"}),
            code_execution(),
            WEATHER,
        ],
    )
    agent.run("hi")
    tools = capture_server.requests[0]["body"]["tools"]
    assert tools[0]["name"] == "get_weather"
    assert tools[1:] == [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 2,
            "blocked_domains": ["x.com"],
            "user_location": {"type": "approximate", "city": "Paris"},
        },
        {"type": "code_execution_20250825", "name": "code_execution"},
    ]


def test_anthropic_turn_goes_back_verbatim_and_the_search_is_recorded(capture_server) -> None:
    _search_then_answer(capture_server)
    agent = Agent(name="a", llm=_claude(capture_server.url), tools=[web_search(), WEATHER])
    result = agent.run("Latest Python, and Paris weather?")
    assert result.output == CITED_TEXT["text"]
    second = capture_server.requests[1]["body"]["messages"]
    assert second[1] == {"role": "assistant", "content": [SEARCH_USE, WEATHER_USE]}
    assert second[2]["content"][0]["type"] == "tool_result"
    (search,) = [c for c in result.tool_calls if c["tool_name"] == "web_search"]
    # Reported with the response that ran it; its query from the turn that asked.
    assert search["arguments"] == {"query": "latest stable Python release"}
    assert "https://www.python.org/downloads/release/python-3150/" in search["output"]
    assert "Python 3.15.0 is the newest major release" in search["output"]
    assert [c["tool_name"] for c in result.tool_calls] == ["get_weather", "web_search"]


def test_anthropic_search_fee_is_priced_with_the_tokens(capture_server) -> None:
    _search_then_answer(capture_server)
    agent = Agent(name="a", llm=_claude(capture_server.url), tools=[web_search(), WEATHER])
    result = agent.run("go")
    tokens = 2 * (1000 * 2.0 + 100 * 10.0) / 1_000_000  # Sonnet 5.5 $2 / $10
    assert result.cost == pytest.approx(tokens + 0.01)  # + $10 per 1,000 searches


def test_anthropic_pause_turn_is_continued(capture_server) -> None:
    capture_server.set_response_sequence(
        [
            (
                200,
                _anthropic(
                    [SEARCH_USE, SEARCH_RESULT, {"type": "text", "text": "Part one. "}],
                    stop="pause_turn",
                    searches=1,
                ),
            ),
            (200, _anthropic([{"type": "text", "text": "Part two."}])),
        ]
    )
    agent = Agent(name="a", llm=_claude(capture_server.url), tools=[web_search()])
    result = agent.run("research")
    assert result.output == "Part one. Part two."
    second = capture_server.requests[1]["body"]["messages"]
    assert second[-1] == {
        "role": "assistant",
        "content": [SEARCH_USE, SEARCH_RESULT, {"type": "text", "text": "Part one. "}],
    }


def test_anthropic_code_execution_is_recorded(capture_server) -> None:
    use = {
        "type": "server_tool_use",
        "id": "srvtoolu_c",
        "name": "bash_code_execution",
        "input": {"command": "python3 -c 'print(832040)'"},
    }
    result_block = {
        "type": "bash_code_execution_tool_result",
        "tool_use_id": "srvtoolu_c",
        "content": {
            "type": "bash_code_execution_result",
            "stdout": "832040\n",
            "stderr": "",
            "return_code": 0,
            "content": [],
        },
    }
    capture_server.set_response_sequence(
        [(200, _anthropic([use, result_block, {"type": "text", "text": "832040"}]))]
    )
    result = Agent(name="a", llm=_claude(capture_server.url), tools=[code_execution()]).run("fib")
    (call,) = result.tool_calls
    assert call["tool_name"] == "code_execution"
    assert "832040" in call["output"] and "python3" in call["output"]


def _sse_anthropic(*events: tuple[str, dict[str, Any]]) -> str:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


@pytest.mark.asyncio
async def test_anthropic_stream_rebuilds_the_turn_for_sending_back(capture_server) -> None:
    turn_1 = _sse_anthropic(
        ("message_start", {"message": {"usage": {"input_tokens": 10}}}),
        ("content_block_start", {"index": 0, "content_block": {**SEARCH_USE, "input": {}}}),
        (
            "content_block_delta",
            {
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(SEARCH_USE["input"]),
                },
            },
        ),
        ("content_block_stop", {"index": 0}),
        ("content_block_start", {"index": 1, "content_block": {**WEATHER_USE, "input": {}}}),
        (
            "content_block_delta",
            {
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"city": "Paris"}'},
            },
        ),
        ("content_block_stop", {"index": 1}),
        ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
    )
    turn_2 = _sse_anthropic(
        ("message_start", {"message": {"usage": {"input_tokens": 20}}}),
        ("content_block_start", {"index": 0, "content_block": SEARCH_RESULT}),
        ("content_block_stop", {"index": 0}),
        ("content_block_start", {"index": 1, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"index": 1, "delta": {"type": "text_delta", "text": "3.15."}}),
        ("content_block_stop", {"index": 1}),
        (
            "message_delta",
            {
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5, "server_tool_use": {"web_search_requests": 1}},
            },
        ),
    )
    capture_server.set_response_sequence(
        [(200, turn_1, "text/event-stream"), (200, turn_2, "text/event-stream")]
    )
    agent = Agent(name="a", llm=_claude(capture_server.url), tools=[web_search(), WEATHER])
    events = [e async for e in agent.astream("go")]
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "3.15."
    assert not any(isinstance(e, ProviderTurn) for e in events)  # consumed by the loop
    second = capture_server.requests[1]["body"]["messages"]
    assert second[1] == {"role": "assistant", "content": [SEARCH_USE, WEATHER_USE]}


def test_the_turn_survives_a_pause_and_resume(capture_server, tmp_path) -> None:
    def approve(city: str) -> str:
        """Ask a human first."""
        decision = interrupt(reason="ok?", context={"city": city})
        return f"approved={decision.approved}"

    use = {**WEATHER_USE, "name": "approve"}
    capture_server.set_response_sequence(
        [
            (200, _anthropic([SEARCH_USE, use], stop="tool_use")),
            (200, _anthropic([SEARCH_RESULT, CITED_TEXT], searches=1)),
        ]
    )
    agent = Agent(
        name="a",
        llm=_claude(capture_server.url),
        tools=[web_search(), FunctionTool(name="approve", fn=approve)],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "cp.db")),
    )
    assert agent.run("go", execution_id="ex-p").status == "paused"
    resumed = agent.resume("ex-p", resume_value=Resume(approved=True))
    assert resumed.output == CITED_TEXT["text"]
    sent = capture_server.requests[-1]["body"]["messages"]
    assert {"role": "assistant", "content": [SEARCH_USE, use]} in sent


def test_checkpoint_keeps_provider_content_only_when_present() -> None:
    plain = AssistantMessage(content="hi")
    carried = AssistantMessage(
        content="hi", provider_content=ProviderContent(provider="anthropic", data=[SEARCH_USE])
    )
    dumped = _serialize_messages([plain, carried])
    assert "provider_content" not in dumped[0]
    assert dumped[1]["provider_content"] == {"provider": "anthropic", "data": [SEARCH_USE]}
    back = _deserialize_messages(dumped)
    assert back[1].content_for("anthropic") == [SEARCH_USE]
    assert back[1].content_for("gemini") is None  # only to the provider that issued it


# ── Gemini ──────────────────────────────────────────────────────────────────

GEMINI_TURN = [
    {
        "toolCall": {
            "toolType": "GOOGLE_SEARCH_WEB",
            "args": {"queries": ["latest stable Python release"]},
            "id": "call_9",
        },
        "thoughtSignature": "sig-a",
    },
    {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}, "id": "call_10"}},
    {
        "toolResponse": {
            "toolType": "GOOGLE_SEARCH_WEB",
            "response": {"search_suggestions": "<div>…</div>"},
            "id": "call_9",
        },
        "thoughtSignature": "sig-b",
    },
]
GROUNDING = {
    "webSearchQueries": ["latest stable Python release"],
    "groundingChunks": [
        {"web": {"uri": "https://vertexaisearch.example/r1", "title": "python.org"}}
    ],
    "groundingSupports": [{"segment": {"text": "Python 3.15.0 is the latest."}}],
}


def _gemini(parts: list[dict[str, Any]], grounding: dict | None = None, **usage: Any) -> dict:
    candidate: dict[str, Any] = {
        "content": {"role": "model", "parts": parts},
        "finishReason": "STOP",
    }
    if grounding:
        candidate["groundingMetadata"] = grounding
    return {
        "candidates": [candidate],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20, **usage},
        "modelVersion": "gemini-3.8-flash",
    }


def _gem(url: str) -> LLMClient:
    return LLMClient(provider="gemini", model="gemini-3.8-flash", api_key="k", base_url=url)


def test_gemini_2_5_and_earlier_refuse_built_ins_beside_functions(capture_server) -> None:
    """Measured: gemini-2.5-flash 400s on the pair ("Tool call context circulation
    is not enabled" with the flag), so it is refused before sending. Alone, a
    built-in tool still works on 2.5."""
    llm = LLMClient(
        provider="gemini", model="gemini-2.5-flash", api_key="k", base_url=capture_server.url
    )
    with pytest.raises(LLMError, match="Gemini 3"):
        Agent(name="a", llm=llm, tools=[web_search(), WEATHER]).run("go")
    assert capture_server.requests == []
    capture_server.set_response_sequence([(200, _gemini([{"text": "3.15"}], GROUNDING))])
    assert Agent(name="a", llm=llm, tools=[web_search()]).run("go").output == "3.15"


def test_gemini_built_ins_beside_functions(capture_server) -> None:
    capture_server.set_response_sequence(
        [
            (200, _gemini(GEMINI_TURN)),
            (200, _gemini([{"text": "Python 3.15.0; 14C in Paris."}], GROUNDING)),
        ]
    )
    agent = Agent(name="a", llm=_gem(capture_server.url), tools=[WEATHER, web_search()])
    result = agent.run("go")
    assert result.output == "Python 3.15.0; 14C in Paris."
    first = capture_server.requests[0]["body"]
    assert first["tools"][1:] == [{"google_search": {}}]
    assert first["toolConfig"] == {
        "includeServerSideToolInvocations": True,
        "functionCallingConfig": {"mode": "VALIDATED"},
    }
    second = capture_server.requests[1]["body"]["contents"]
    assert second[1] == {"role": "model", "parts": GEMINI_TURN}
    response_part = second[2]["parts"][0]["functionResponse"]
    assert response_part["id"] == "call_10" and response_part["name"] == "get_weather"
    (search,) = [c for c in result.tool_calls if c["tool_name"] == "web_search"]
    assert search["arguments"] == {"queries": ["latest stable Python release"]}
    assert "python.org" in search["output"] and "Python 3.15.0 is the latest." in search["output"]


def test_gemini_search_fee_and_tool_use_tokens_are_priced(capture_server) -> None:
    capture_server.set_response_sequence(
        [(200, _gemini([{"text": "3.15"}], GROUNDING, toolUsePromptTokenCount=900))]
    )
    llm = _gem(capture_server.url)
    response = llm.complete([UserMessage("go")], tools=[web_search().to_openai_format()])
    assert capture_server.requests[0]["body"]["tools"] == [{"google_search": {}}]
    assert "toolConfig" not in capture_server.requests[0]["body"]
    assert response.usage["prompt_tokens"] == 1000  # 100 + 900 tool-use prompt
    assert response.usage["tool_requests"] == {"web_search": 1}
    usd, known = tool_fees_cost("gemini", response.usage)
    assert known and usd == pytest.approx(0.014)


def test_gemini_code_execution_is_recorded(capture_server) -> None:
    parts = [
        {"executableCode": {"language": "PYTHON", "code": "print(832040)", "id": "c1"}},
        {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "832040\n", "id": "c1"}},
        {"text": "832040"},
    ]
    capture_server.set_response_sequence([(200, _gemini(parts))])
    result = Agent(name="a", llm=_gem(capture_server.url), tools=[code_execution()]).run("fib")
    (call,) = result.tool_calls
    assert call["tool_name"] == "code_execution" and "832040" in call["output"]
    assert capture_server.requests[0]["body"]["tools"] == [{"code_execution": {}}]


# ── OpenAI Responses ────────────────────────────────────────────────────────

WS_CALL = {
    "id": "ws_1",
    "type": "web_search_call",
    "status": "completed",
    "action": {
        "type": "search",
        "query": "latest stable Python release",
        "queries": ["latest stable Python release"],
        "sources": [
            {"type": "url", "url": "https://www.python.org/downloads/release/python-3150/"}
        ],
    },
}
FN_CALL = {
    "id": "fc_1",
    "type": "function_call",
    "status": "completed",
    "call_id": "call_1",
    "name": "get_weather",
    "arguments": '{"city": "Paris"}',
}
CI_CALL = {
    "id": "ci_1",
    "type": "code_interpreter_call",
    "status": "completed",
    "code": "print(832040)",
    "container_id": "cntr_1",
    "outputs": [{"type": "logs", "logs": "832040"}],
}


def _responses(output: list[dict[str, Any]], searches: int = 0) -> dict:
    return {
        "id": "r",
        "status": "completed",
        "model": "gpt-5.1",
        "output": output,
        "usage": {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100},
        "tool_usage": {"web_search": {"num_requests": searches}},
    }


def _msg(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _gpt(url: str) -> LLMClient:
    return LLMClient(provider="openai", model="gpt-5.1", api_key="k", base_url=url, api="responses")


def test_responses_built_ins_and_the_turn_back(capture_server) -> None:
    capture_server.set_response_sequence(
        [
            (200, _responses([WS_CALL, FN_CALL], searches=1)),
            (200, _responses([_msg("3.15; 14C in Paris.")])),
        ]
    )
    tools = [
        web_search(allowed_domains=["python.org"]),
        file_search(["vs_1"], max_results=3),
        WEATHER,
    ]
    result = Agent(name="a", llm=_gpt(capture_server.url), tools=tools).run("go")
    assert result.output == "3.15; 14C in Paris."
    body = capture_server.requests[0]["body"]
    assert {"type": "web_search", "filters": {"allowed_domains": ["python.org"]}} in body["tools"]
    assert {"type": "file_search", "vector_store_ids": ["vs_1"], "max_num_results": 3} in body[
        "tools"
    ]
    assert "web_search_call.action.sources" in body["include"]
    second = capture_server.requests[1]["body"]["input"]
    assert second[1:3] == [WS_CALL, FN_CALL]  # verbatim, with the provider's ids
    (search,) = [c for c in result.tool_calls if c["tool_name"] == "web_search"]
    assert "python.org/downloads/release/python-3150" in search["output"]


def test_responses_container_is_billed_once_per_run(capture_server) -> None:
    capture_server.set_response_sequence(
        [
            (200, _responses([CI_CALL, FN_CALL])),
            (200, _responses([{**CI_CALL, "id": "ci_2"}, _msg("done")])),
        ]
    )
    agent = Agent(name="a", llm=_gpt(capture_server.url), tools=[code_execution(), WEATHER])
    result = agent.run("go")
    tokens = 2 * (1000 * 1.25 + 100 * 10.0) / 1_000_000  # gpt-5.1
    assert result.cost == pytest.approx(tokens + 0.03)  # one container, two calls
    assert [c["tool_name"] for c in result.tool_calls].count("code_execution") == 2


# ── guardrails ──────────────────────────────────────────────────────────────


def _rule(position: GuardrailPosition, fn: Any, **kw: Any) -> Guardrail:
    return Guardrail(
        name=f"rule_{position.value}",
        guardrail_type=GuardrailType.code,
        position=position,
        blocking=True,
        fn=fn,
        **kw,
    )


def test_tool_call_rule_blocks_before_the_model_is_called(capture_server) -> None:
    rule = _rule(GuardrailPosition.tool_call, lambda text: GuardrailResult(passed=True))
    agent = Agent(
        name="a", llm=_claude(capture_server.url), tools=[web_search()], guardrails=[rule]
    )
    with pytest.raises(GuardrailBlockedError) as err:
        agent.run("go")
    assert capture_server.requests == []  # nothing searched, nothing spent
    (result,) = err.value.results
    assert result.errored and not result.passed
    assert "run by the model's provider" in result.message


def test_tool_call_rule_on_error_allow_records_each_call_unchecked(capture_server) -> None:
    rule = _rule(
        GuardrailPosition.tool_call, lambda text: GuardrailResult(passed=True), on_error="allow"
    )
    _search_then_answer(capture_server)
    agent = Agent(
        name="a", llm=_claude(capture_server.url), tools=[web_search(), WEATHER], guardrails=[rule]
    )
    result = agent.run("go")
    firings = [g for g in result.guardrails if g.position == "tool_call"]
    # get_weather was really checked; the search could not be.
    assert [f.effect for f in firings] == ["none", "unchecked"]


def test_tool_result_rule_judges_what_the_provider_returned(capture_server) -> None:
    seen: list[str] = []

    def no_python_org(text: str) -> GuardrailResult:
        seen.append(text)
        return GuardrailResult(passed="python.org" not in text, message="source not allowed")

    _search_then_answer(capture_server)
    rule = _rule(GuardrailPosition.tool_result, no_python_org)
    agent = Agent(
        name="a", llm=_claude(capture_server.url), tools=[web_search(), WEATHER], guardrails=[rule]
    )
    with pytest.raises(GuardrailBlockedError, match="source not allowed"):
        agent.run("go")
    assert any("query: latest stable Python release" in s for s in seen)


def test_a_rewrite_of_a_provider_result_blocks(capture_server) -> None:
    _search_then_answer(capture_server)
    rule = _rule(
        GuardrailPosition.tool_result,
        lambda text: GuardrailResult(passed=False, message="mask it"),
        action="override",
        config={"override_with": "[removed]"},
    )
    agent = Agent(
        name="a", llm=_claude(capture_server.url), tools=[web_search(), WEATHER], guardrails=[rule]
    )
    with pytest.raises(GuardrailBlockedError, match="already read"):
        agent.run("go")


def test_a_result_with_nothing_to_judge_is_could_not_run(capture_server) -> None:
    empty = {**SEARCH_RESULT, "content": []}
    use = {**SEARCH_USE, "input": {"query": ""}}
    capture_server.set_response_sequence(
        [(200, _anthropic([use, empty, {"type": "text", "text": "nothing"}]))]
    )
    calls: list[str] = []
    rule = _rule(
        GuardrailPosition.tool_result,
        lambda text: calls.append(text) or GuardrailResult(passed=True),
    )
    agent = Agent(
        name="a", llm=_claude(capture_server.url), tools=[web_search()], guardrails=[rule]
    )
    with pytest.raises(GuardrailBlockedError):
        agent.run("go")
    assert calls == []  # never judged as a clean, empty pass


# ── spans ───────────────────────────────────────────────────────────────────


def test_tool_span_is_recorded_and_its_payload_gated(capture_server, monkeypatch, tmp_path) -> None:
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    otel.reset()
    try:
        for payloads in ("1", "0"):
            monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", payloads)
            local, egress = InMemorySpanExporter(), InMemorySpanExporter()
            otel.add_span_processor(SimpleSpanProcessor(local))
            otel.add_exporter(egress)
            _search_then_answer(capture_server)
            Agent(name="a", llm=_claude(capture_server.url), tools=[web_search(), WEATHER]).run(
                "go"
            )
            otel.get_tracer_provider().force_flush()
            (span,) = [s for s in local.get_finished_spans() if s.name == "tool.web_search"]
            attrs = dict(span.attributes)
            assert attrs["tool.origin"] == "provider"
            assert attrs["fastaiagent.runner.type"] == "tool"
            assert attrs["fastaiagent.tool.replay_class"] == "side_effecting"
            assert attrs["tool.status"] == "ok"
            # Local capture is full fidelity; the payload switch gates egress.
            assert "python.org" in attrs["tool.result"]
            (out,) = [s for s in egress.get_finished_spans() if s.name == "tool.web_search"]
            exported = dict(out.attributes)
            assert exported["tool.origin"] == "provider"
            if payloads == "1":
                assert "python.org" in exported["tool.result"]
            else:
                assert "tool.result" not in exported and "tool.args" not in exported
                assert "python.org" not in json.dumps(exported, default=str)
            otel.reset()
    finally:
        otel.reset()
        reset_config()


# ── containers and pricing ──────────────────────────────────────────────────


def test_in_a_chain_node(capture_server) -> None:
    _search_then_answer(capture_server)
    agent = Agent(name="a", llm=_claude(capture_server.url), tools=[web_search(), WEATHER])
    chain = Chain("c", checkpoint_enabled=False)
    chain.add_node("ask", agent=agent)
    result = chain.execute({"message": "go"})
    assert CITED_TEXT["text"] in json.dumps(result.output, default=str)
    second = capture_server.requests[1]["body"]["messages"]
    assert second[1] == {"role": "assistant", "content": [SEARCH_USE, WEATHER_USE]}


@pytest.mark.asyncio
async def test_in_a_swarm(capture_server) -> None:
    from fastaiagent import Swarm

    _search_then_answer(capture_server)
    agent = Agent(name="a", llm=_claude(capture_server.url), tools=[web_search(), WEATHER])
    swarm = Swarm(name="s", agents=[agent], entrypoint="a", handoffs={"a": []})
    result = await swarm.arun("go")
    assert result.output == CITED_TEXT["text"]
    second = capture_server.requests[1]["body"]["messages"]
    assert {"role": "assistant", "content": [SEARCH_USE, WEATHER_USE]} in second


@pytest.mark.asyncio
async def test_as_a_supervisor_worker(capture_server) -> None:
    from fastaiagent import Supervisor, Worker
    from fastaiagent.llm.client import LLMResponse
    from fastaiagent.llm.message import ToolCall
    from tests.conftest import MockLLMClient

    _search_then_answer(capture_server)
    worker = Agent(name="w", llm=_claude(capture_server.url), tools=[web_search(), WEATHER])
    lead = MockLLMClient(
        responses=[
            LLMResponse(
                tool_calls=[ToolCall(id="s1", name="delegate_to_w", arguments={"task": "go"})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="lead done", finish_reason="stop"),
        ]
    )
    sup = Supervisor(name="lead", llm=lead, workers=[Worker(agent=worker)])
    result = await sup.arun("go")
    assert result.output == "lead done"
    second = capture_server.requests[1]["body"]["messages"]
    assert {"role": "assistant", "content": [SEARCH_USE, WEATHER_USE]} in second


def test_tool_fees_unknown_provider_and_overrides() -> None:
    usage = {"tool_requests": {"web_search": 3}}
    assert tool_fees_cost("openai", usage) == (pytest.approx(0.03), True)
    assert tool_fees_cost("azure", usage) == (0.0, False)  # unpriced, never guessed
    assert tool_fees_cost("openai", {}) == (0.0, True)
    set_tool_fee_overrides({"openai/web_search": 0.005, "azure/web_search": 0.01})
    try:
        assert tool_fees_cost("openai", usage)[0] == pytest.approx(0.015)
        assert tool_fees_cost("azure", usage) == (pytest.approx(0.03), True)
    finally:
        set_tool_fee_overrides(None)
    with pytest.raises(ValueError):
        set_tool_fee_overrides({"web_search": 1.0})


def test_an_unpriced_fee_makes_the_run_cost_unknown() -> None:
    from fastaiagent._internal.pricing import record_run_cost

    token = start_run_cost()
    try:
        record_run_cost(
            "gpt-5.1",
            {"prompt_tokens": 10, "completion_tokens": 1, "tool_requests": {"web_search": 1}},
            provider="azure",
        )
        assert run_cost() == (0.0, False)
    finally:
        stop_run_cost(token)
