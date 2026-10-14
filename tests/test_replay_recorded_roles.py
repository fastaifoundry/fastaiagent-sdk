"""Recorded replay: a guardrail's model call never takes an agent turn (1.89.0).

``with_determinism("recorded")`` serves the original trace's captured model
responses in order, one per ``LLMClient.acomplete``. A model-backed guardrail
(``llm_judge`` here) makes its own ``acomplete``, and until 1.89.0 it popped
from the same queue:

* a judge in the **rerun only** (a plane-distributed rule added after the
  original run) took the agent's first turn, so the rerun was blocked or every
  later turn got the wrong response;
* a judge in the **original only** left its verdict at the front of the queue,
  and the agent's first turn answered with it.

Now each call is served from its own role's recordings (agent / guardrail),
and a judge with no recording is a miss under ``on_miss``.

No mocks: a real ``Agent`` with a real ``LLMClient`` talks to a local
OpenAI-compatible HTTP server that answers the judge and the agent apart; the
judge is a real plane rule (``policy_cache``, as ``connect()`` fills it) built
by the real ``from_policy`` builder; the original trace is the real one in
``local.db``.

Sweep: judge in the rerun only / in both / in neither / in the original only,
the judge on the input and on the output side, ``on_miss`` live and error, and
a trace recorded before 1.89.0 (the judge's ``llm.*`` span beside, not under,
its guardrail span).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

AGENT_NAME = "roles-agent"
FINAL = "The answer is 42."
TOOL_TURN = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_lookup",
            "type": "function",
            "function": {"name": "lookup", "arguments": json.dumps({"q": "answer"})},
        }
    ],
}


class _ChatServer:
    """OpenAI ``/chat/completions``: the judge gets a PASS verdict, the agent its script."""

    def __init__(self) -> None:
        self.judge_calls = 0
        self.agent_calls = 0
        self._server: HTTPServer | None = None

    def reset(self) -> None:
        self.judge_calls = 0
        self.agent_calls = 0

    def reply(self, body: dict[str, Any]) -> dict[str, Any]:
        msgs = body.get("messages") or []
        system = next((m.get("content") for m in msgs if m.get("role") == "system"), "") or ""
        if str(system).startswith("You are a guardrail judge"):
            self.judge_calls += 1
            message: dict[str, Any] = {
                "role": "assistant",
                "content": json.dumps({"verdict": "PASS", "reason": "fine"}),
            }
        else:
            self.agent_calls += 1
            has_tool_result = any(m.get("role") == "tool" for m in msgs)
            message = {"role": "assistant", "content": FINAL} if has_tool_result else TOOL_TURN
        return {
            "id": "chatcmpl-roles",
            "object": "chat.completion",
            "created": 1700000000,
            "model": body.get("model", "gpt-4o-mini"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def start(self) -> _ChatServer:
        chat = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                raw = json.dumps(chat.reply(body)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args: Any) -> None:
                return

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    @property
    def base_url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


@pytest.fixture(scope="module")
def chat() -> Any:
    server = _ChatServer().start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chat: Any) -> Any:
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.client import _connection
    from fastaiagent.guardrail.from_policy import clear_cache
    from fastaiagent.trace import otel

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    monkeypatch.setenv("FASTAIAGENT_ALLOW_PRIVATE_NETWORKS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "stub-key-for-local-server")
    for name in (
        "FASTAIAGENT_API_KEY",
        "FASTAIAGENT_TARGET",
        "FASTAIAGENT_EXPORT_EXPERIMENTS",
        "FASTAIAGENT_REPLAY_TOOL_POLICY",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    _project.set_project_id("test-proj")
    otel.reset()
    saved = _connection.policy_cache
    chat.reset()
    try:
        yield chat
    finally:
        _connection.policy_cache = saved
        clear_cache()
        otel.reset()
        _project.reset_for_testing()
        reset_config()


def _plane_judge(chat: Any, *, position: str, on_error: str = "block") -> None:
    """Distribute one domain-wide llm_judge rule, as ``connect()`` would."""
    from fastaiagent.client import _connection
    from fastaiagent.guardrail.from_policy import clear_cache

    _connection.policy_cache = {
        "version": f"v-{position}-{on_error}",
        "guardrail_rules": [
            {
                "name": f"plane-judge-{position}",
                "guardrail_type": position,
                "validation_mode": "blocking",
                "implementation_type": "llm_judge",
                "config": {
                    "prompt": "Is this a polite request?",
                    "llm": {
                        "provider": "custom",
                        "model": "gpt-4o-mini",
                        "base_url": chat.base_url,
                    },
                },
                "on_error": on_error,
                "agent_ids": [],
            }
        ],
        "approval_policies": [],
    }
    clear_cache()


def _no_plane() -> None:
    from fastaiagent.client import _connection
    from fastaiagent.guardrail.from_policy import clear_cache

    _connection.policy_cache = None
    clear_cache()


def _original_run(chat: Any) -> str:
    import fastaiagent as fa
    from fastaiagent.llm import LLMClient

    agent = fa.Agent(
        name=AGENT_NAME,
        system_prompt="You answer questions.",
        llm=LLMClient(provider="custom", model="gpt-4o-mini", base_url=chat.base_url),
        tools=[_lookup_tool()],
    )
    result = agent.run("What is the answer?")
    assert result.output == FINAL
    return result.trace_id


def _lookup_tool() -> Any:
    import fastaiagent as fa

    def lookup(q: str) -> str:
        """Look the answer up."""
        return "42"

    return fa.FunctionTool(name="lookup", fn=lookup)


def _rerun(trace_id: str, *, on_miss: str = "error") -> Any:
    from fastaiagent.trace.replay import Replay

    return (
        Replay.load(trace_id)
        .fork_at(0)
        .with_tool_override("lookup", _lookup_tool())
        .with_determinism("recorded", on_miss=on_miss)
        .rerun()
    )


def _rerun_llm_spans(result: Any) -> list[Any]:
    from fastaiagent.trace.storage import TraceStore

    spans = TraceStore.default().get_trace(result.trace_id).spans
    return [s for s in spans if s.name.startswith("llm.")]


# -- the judge is in the rerun only -------------------------------------------------


@pytest.mark.parametrize("position", ["input", "output"])
def test_a_rerun_only_judge_misses_under_on_miss_live_and_takes_no_agent_turn(
    world: Any, position: str
) -> None:
    trace_id = _original_run(world)
    _plane_judge(world, position=position)
    world.reset()

    result = _rerun(trace_id, on_miss="live")

    assert result.new_output == FINAL
    assert world.judge_calls == 1, "the judge's miss should be one live judge call"
    assert world.agent_calls == 0, "every agent turn must come from the recording"
    recorded = [s for s in _rerun_llm_spans(result) if s.attributes.get("replay.mode")]
    assert len(recorded) == 2, "both agent turns are served from the recording"


def test_a_rerun_only_judge_under_on_miss_error_follows_its_on_error(world: Any) -> None:
    """``on_miss="error"``: the judge's miss raises inside the check, so the rule's
    own ``on_error`` decides. With ``on_error="allow"`` the agent then runs on
    its recorded turns, untouched; nothing goes live."""
    trace_id = _original_run(world)
    _plane_judge(world, position="input", on_error="allow")
    world.reset()

    result = _rerun(trace_id, on_miss="error")

    assert result.new_output == FINAL
    assert (world.judge_calls, world.agent_calls) == (0, 0)


def test_a_rerun_only_judge_with_on_error_block_blocks_rather_than_stealing(world: Any) -> None:
    from fastaiagent._internal.errors import GuardrailBlockedError

    trace_id = _original_run(world)
    _plane_judge(world, position="input", on_error="block")
    world.reset()

    with pytest.raises(GuardrailBlockedError) as caught:
        _rerun(trace_id, on_miss="error")
    assert (world.judge_calls, world.agent_calls) == (0, 0)
    # Blocked because the judge could not run (its miss raised), not because it
    # judged the agent's recorded tool-call turn and failed it.
    results = caught.value.results
    assert results and all(getattr(r, "errored", False) for r in results), results
    assert any("ran out of captured LLM responses" in (r.message or "") for r in results)


# -- the judge is in both / neither / the original only -----------------------------


@pytest.mark.parametrize("position", ["input", "output"])
def test_a_judge_in_both_runs_is_served_its_own_recording(world: Any, position: str) -> None:
    _plane_judge(world, position=position)
    trace_id = _original_run(world)
    assert world.judge_calls == 1
    world.reset()

    result = _rerun(trace_id, on_miss="error")

    assert result.new_output == FINAL
    assert (world.judge_calls, world.agent_calls) == (0, 0)
    assert len([s for s in _rerun_llm_spans(result) if s.attributes.get("replay.mode")]) == 3


def test_no_judge_anywhere_replays_as_before(world: Any) -> None:
    _no_plane()
    trace_id = _original_run(world)
    world.reset()

    result = _rerun(trace_id, on_miss="error")

    assert result.new_output == FINAL
    assert (world.judge_calls, world.agent_calls) == (0, 0)


@pytest.mark.parametrize("position", ["input", "output"])
def test_a_judge_in_the_original_only_never_answers_for_the_agent(
    world: Any, position: str
) -> None:
    _plane_judge(world, position=position)
    trace_id = _original_run(world)
    _no_plane()
    world.reset()

    result = _rerun(trace_id, on_miss="error")

    assert result.new_output == FINAL, "the agent was handed the judge's verdict"
    assert (world.judge_calls, world.agent_calls) == (0, 0)


# -- a trace recorded before 1.89.0 ---------------------------------------------------


def _as_pre_189(trace: Any) -> Any:
    """The 1.88.0 shape: the guardrail span did not wrap its check, so the
    judge's ``llm.*`` span was its sibling, not its child."""
    by_id = {s.span_id: s for s in trace.spans}
    spans = []
    for s in trace.spans:
        parent = by_id.get(s.parent_span_id or "")
        if (
            s.name.startswith("llm.")
            and parent is not None
            and parent.name.startswith("guardrail.")
        ):
            s = s.model_copy(update={"parent_span_id": parent.parent_span_id})
        spans.append(s)
    by_new = {s.span_id: s for s in spans}
    assert not any(
        s.name.startswith("llm.")
        and (by_new.get(s.parent_span_id or "") or s).name.startswith("guardrail.")
        for s in spans
    ), "a judge span is still under its guardrail span"
    assert any(s.name.startswith("guardrail.") for s in spans)
    return trace.model_copy(update={"spans": spans})


def test_a_pre_189_trace_still_tells_the_judge_from_the_agent(world: Any) -> None:
    from fastaiagent.trace.replay import Replay
    from fastaiagent.trace.storage import TraceStore

    _plane_judge(world, position="input")
    trace_id = _original_run(world)
    _no_plane()
    world.reset()

    old = _as_pre_189(TraceStore.default().get_trace(trace_id))
    result = (
        Replay(old)
        .fork_at(0)
        .with_tool_override("lookup", _lookup_tool())
        .with_determinism("recorded", on_miss="error")
        .rerun()
    )
    assert result.new_output == FINAL
    assert (world.judge_calls, world.agent_calls) == (0, 0)


def test_with_tracing_off_the_rerun_falls_back_to_one_queue_and_says_so(
    world: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No spans, no role to read: the pre-1.89.0 shared queue, with a warning."""
    import logging

    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel
    from fastaiagent.trace.replay import Replay

    _no_plane()
    trace_id = _original_run(world)
    forked = (
        Replay.load(trace_id)
        .fork_at(0)
        .with_tool_override("lookup", _lookup_tool())
        .with_determinism("recorded", on_miss="error")
    )
    monkeypatch.setenv("FASTAIAGENT_TRACE_ENABLED", "0")
    reset_config()
    otel.reset()
    world.reset()
    with caplog.at_level(logging.WARNING, logger="fastaiagent.trace.replay"):
        result = forked.rerun()
    assert result.new_output == FINAL
    assert (world.judge_calls, world.agent_calls) == (0, 0)
    assert any("tracing is off" in r.message for r in caplog.records)


def test_recorded_roles_split_by_ancestry(world: Any) -> None:
    from fastaiagent.trace._replay_roles import AGENT, GUARDRAIL, split_by_role
    from fastaiagent.trace.replay import _recorded_response_from_span
    from fastaiagent.trace.storage import TraceStore

    _plane_judge(world, position="input")
    trace_id = _original_run(world)
    spans = TraceStore.default().get_trace(trace_id).spans
    split = split_by_role(spans, _recorded_response_from_span)
    assert [r.content for r in split[GUARDRAIL]] == [
        json.dumps({"verdict": "PASS", "reason": "fine"})
    ]
    assert [bool(r.tool_calls) for r in split[AGENT]] == [True, False]
