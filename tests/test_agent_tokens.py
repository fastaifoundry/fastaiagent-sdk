"""``tokens_used`` is the whole run, not the last turn of it.

``_arun_core`` computed ``tokens = response.usage["total_tokens"] + retry_tokens``
and ``execute_tool_loop`` returns only the LAST response — so a run that made
three LLM calls reported the third one's tokens. 1.67.0 made ``cost`` correct
across every turn by accumulating inside ``LLMClient`` through a run-scoped
ContextVar, and the two numbers then visibly disagreed on the same run: a
correct cost next to an understated token count derived from the same completions.

The fix is the same shape as the cost fix — one accumulator, written at the one
place every completion passes through — so the two can never drift apart again.

**No mocks.** These drive a real :class:`~fastaiagent.llm.client.LLMClient` over
httpx against a tiny in-process stub HTTP server, the pattern
``tests/test_agent_cost.py`` established. That matters here more than usual:
tokens are now accumulated *inside* ``LLMClient._acomplete_with_retries``, so a
fake client that replaces ``acomplete`` wholesale would test nothing. The stub
needs no API key and no network.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from fastaiagent.agent import Agent
from fastaiagent.llm.client import LLMClient
from fastaiagent.tool import tool

#: Every stub reply bills this many tokens, split 6 in / 4 out.
PROMPT_T = 6
COMPLETION_T = 4
PER_CALL = PROMPT_T + COMPLETION_T

#: Bodies the stub hands out, in order; the last one repeats.
_REPLIES: list[dict[str, Any]] = []
_CALLS = {"n": 0}


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's API
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        idx = min(_CALLS["n"], len(_REPLIES) - 1)
        _CALLS["n"] += 1
        message = dict(_REPLIES[idx])
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        body = {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "stub",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": PROMPT_T,
                "completion_tokens": COMPLETION_T,
                "total_tokens": PER_CALL,
            },
        }
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: Any) -> None:  # silence the test log
        return


@pytest.fixture
def stub():
    """A local OpenAI-compatible endpoint. Yields a ``configure`` callable."""
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"

    def configure(replies: list[dict[str, Any]] | None = None) -> LLMClient:
        _REPLIES[:] = replies or [{"role": "assistant", "content": "reply"}]
        _CALLS["n"] = 0
        return LLMClient(
            provider="custom",
            model="gpt-4o-mini",
            base_url=base_url,
            api_key="stub",
            max_retries=0,
        )

    try:
        yield configure
    finally:
        server.shutdown()
        server.server_close()


def _text(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def _calls_tool(name: str, args: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def _cp(tmp_path):
    from fastaiagent import SQLiteCheckpointer

    cp = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    cp.setup()
    return cp


@tool()
def lookup(q: str) -> str:
    """Look something up."""
    return f"found:{q}"


@tool()
def second(q: str) -> str:
    """Look something else up."""
    return f"also:{q}"


# ---------------------------------------------------------------------------
# The number itself
# ---------------------------------------------------------------------------


def test_a_multi_turn_tool_loop_bills_every_turn(stub) -> None:
    """Three LLM calls, three billed turns — not just the last one."""
    llm = stub(
        [
            _calls_tool("lookup", {"q": "a"}),
            _calls_tool("second", {"q": "b"}, call_id="call_2"),
            _text("final answer"),
        ]
    )
    result = Agent(name="tok-loop", llm=llm, tools=[lookup, second]).run("go")
    assert result.output == "final answer"
    assert _CALLS["n"] == 3, f"expected three LLM calls, saw {_CALLS['n']}"
    assert result.tokens_used == 3 * PER_CALL, (
        f"tokens_used is {result.tokens_used}; the tool loop returns only the last "
        f"response, so the first two turns went unbilled"
    )


def test_a_single_turn_run_is_unchanged(stub) -> None:
    """The fix must not inflate the shape that was already right."""
    llm = stub([_text("reply")])
    result = Agent(name="tok-single", llm=llm).run("hi")
    assert result.tokens_used == PER_CALL


def test_the_structured_reask_turn_is_not_double_counted(stub) -> None:
    """``_reask_structured`` already added its own ``retry_tokens`` at the call
    site. Accumulating at the client and then *also* adding that number would
    bill the re-ask twice — the opposite error, and just as wrong."""
    from pydantic import BaseModel

    class Out(BaseModel):
        note: str

    llm = stub([_text("not json at all"), _text('{"note": "ok"}')])
    result = Agent(name="tok-reask", llm=llm, output_type=Out).run("hi")
    assert result.parsed is not None, "the re-ask did not fire — test no longer covers it"
    assert _CALLS["n"] == 2, f"expected two LLM calls, saw {_CALLS['n']}"
    assert result.tokens_used == 2 * PER_CALL, (
        f"tokens_used is {result.tokens_used}, not {2 * PER_CALL}: the re-ask turn is "
        "counted twice (or not at all)"
    )


def test_a_guardrail_reask_turn_is_billed_once(stub) -> None:
    """The output guardrail's re-ask is the third call site that used to hand a
    token count back up by hand."""
    from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType

    rule = Guardrail(
        name="no-digits",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": r"\d{3}-\d{2}-\d{4}"},
        action="reask",
    )
    llm = stub([_text("ssn 123-45-6789"), _text("clean answer")])
    result = Agent(name="tok-guard", llm=llm, guardrails=[rule]).run("go")
    assert result.output == "clean answer"
    assert _CALLS["n"] == 2, f"expected two LLM calls, saw {_CALLS['n']}"
    assert result.tokens_used == 2 * PER_CALL


def test_cost_and_tokens_agree_on_the_same_run(stub) -> None:
    """Their disagreement is the symptom. Both are derived from the same set of
    completions, so a run whose cost covers three turns must have a token count
    that covers three turns."""
    from fastaiagent._internal.pricing import compute_cost_usd

    llm = stub(
        [
            _calls_tool("lookup", {"q": "a"}),
            _calls_tool("second", {"q": "b"}, call_id="call_2"),
            _text("done"),
        ]
    )
    result = Agent(name="tok-vs-cost", llm=llm, tools=[lookup, second]).run("go")
    assert result.cost_known is True
    expected_cost = compute_cost_usd("gpt-4o-mini", 3 * PROMPT_T, 3 * COMPLETION_T)
    assert result.cost == pytest.approx(expected_cost, rel=1e-9)
    assert result.tokens_used == 3 * PER_CALL, (
        f"cost covers 3 turns but tokens_used ({result.tokens_used}) covers "
        f"{result.tokens_used / PER_CALL:.0f}"
    )


def test_a_paused_run_reports_what_it_already_spent(stub, tmp_path) -> None:
    """A pause is not a reason to report zero — the same reasoning that gave the
    paused branch a real ``cost`` in 1.67.0."""
    from fastaiagent.chain.interrupt import interrupt

    @tool()
    def approve(amount: int) -> str:
        """Ask a human."""
        decision = interrupt(reason="approve?", context={"amount": amount})
        return str(decision)

    llm = stub([_calls_tool("approve", {"amount": 5}), _text("done")])
    agent = Agent(
        name="tok-paused",
        llm=llm,
        tools=[approve],
        checkpointer=_cp(tmp_path),
    )
    result = agent.run("go", execution_id="tok-paused-1")
    assert result.status == "paused"
    assert result.tokens_used == PER_CALL, (
        f"a paused run reported {result.tokens_used} tokens after one billed turn"
    )


# ---------------------------------------------------------------------------
# Nesting: a child's tokens belong to the child
# ---------------------------------------------------------------------------


def test_a_swarm_sums_its_children_without_double_counting(stub) -> None:
    """Each agent opens its own accumulator, so a hop's tokens are reported on
    the hop's result and summed once by the swarm."""
    from fastaiagent.agent import Swarm

    llm = stub([_text("alpha done")])
    alpha = Agent(name="alpha", llm=llm)
    swarm = Swarm(name="solo", agents=[alpha], entrypoint="alpha")
    result = swarm.run("hi")
    assert result.tokens_used == PER_CALL, result.tokens_used
