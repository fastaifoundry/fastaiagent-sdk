"""Live: the Guardrail Events record tells the truth about a real model's runs (1.89.0).

gpt-4o-mini agents, ``FASTAIAGENT_UI_ENABLED`` unset, read back through the
Local UI's own API:

* rows exist without any UI flag;
* a ``reask`` rule that the model satisfies on its retry leaves attempt 1
  ``reasked`` and attempt 2 a pass;
* an observe-only ``mask`` over a tool result is ``observed`` / ``warned``, and
  the tool row carries the agent's name;
* an observe-only rule that cannot run is ``errored`` / ``observed``;
* a streamed reply a ``mask`` rule fires on is ``blocked`` with its reason;
* the Traces status filter and Home's failing count read the root span.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("fastapi")

if not os.environ.get("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY not set", allow_module_level=True)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastapi.testclient import TestClient

    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel
    from fastaiagent.ui import events
    from fastaiagent.ui.server import build_app

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    for name in (
        "FASTAIAGENT_UI_ENABLED",
        "FASTAIAGENT_TRACE_ENABLED",
        "FASTAIAGENT_API_KEY",
        "FASTAIAGENT_TARGET",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    otel.reset()
    events._reset_writers()
    yield db, TestClient(build_app(db_path=str(db), no_auth=True))
    otel.reset()
    reset_config()


def _llm() -> Any:
    from fastaiagent import LLMClient

    return LLMClient(provider="openai", model="gpt-4o-mini", temperature=0)


def _events(client: Any, **params: Any) -> list[dict[str, Any]]:
    r = client.get("/api/guardrail-events", params={"page_size": 200, **params})
    assert r.status_code == 200, r.text
    return r.json()["rows"]


def test_the_record_matches_what_the_runs_did(world: Any) -> None:
    import fastaiagent as fa
    from fastaiagent.guardrail.guardrail import (
        Guardrail,
        GuardrailPosition,
        GuardrailResult,
        GuardrailType,
    )
    from fastaiagent.tool.function import FunctionTool

    db, client = world

    # 1 · A re-ask the model satisfies on its retry.
    def says_banana(text: str) -> GuardrailResult:
        ok = "banana" in text.lower()
        return GuardrailResult(
            passed=ok, message=None if ok else "The reply must contain the word banana."
        )

    reasker = fa.Agent(
        name="fruit",
        system_prompt="Answer in one short sentence.",
        llm=_llm(),
        guardrails=[
            Guardrail(
                name="needs_banana",
                guardrail_type=GuardrailType.code,
                position=GuardrailPosition.output,
                fn=says_banana,
                action="reask",
            )
        ],
    )
    reasked = reasker.run("Name a yellow fruit other than a banana.")
    rows = sorted(
        _events(client, rule="needs_banana", hide_passes=False), key=lambda e: e["attempt"]
    )
    assert [r["attempt"] for r in rows] == [1, 2], rows
    assert (rows[0]["outcome"], rows[0]["effect"]) == ("reasked", "reasked")
    assert rows[1]["outcome"] == "passed" and "banana" in reasked.output.lower()

    # 2 · An observe-only mask over a tool result, and an observer that cannot run.
    def card_on_file(customer: str) -> str:
        """Look up the card on file for a customer."""
        return f"{customer}: card 4111-1111-1111-1111, expires 09/28"

    def boom(_text: str) -> bool:
        raise RuntimeError("detector down")

    billing = fa.Agent(
        name="billing",
        system_prompt="Use card_on_file, then say only whether a card is on file.",
        llm=_llm(),
        tools=[FunctionTool(name="card_on_file", fn=card_on_file, register=False)],
        guardrails=[
            Guardrail(
                name="card_mask_observer",
                guardrail_type=GuardrailType.regex,
                position=GuardrailPosition.tool_result,
                config={"pattern": r"\d{4}-\d{4}-\d{4}-\d{4}"},
                action="mask",
                blocking=False,
            ),
            Guardrail(
                name="flaky_observer",
                guardrail_type=GuardrailType.code,
                position=GuardrailPosition.output,
                fn=boom,
                blocking=False,
            ),
        ],
    )
    billed = billing.run("Is there a card on file for customer Dana?")
    (mask_row,) = _events(client, rule="card_mask_observer")
    assert (mask_row["outcome"], mask_row["effect"]) == ("warned", "observed")
    assert mask_row["agent_name"] == "billing", "a tool-position row names its agent"
    assert mask_row["blocking"] is False and mask_row["action_taken"] == "masked"
    (flaky_row,) = _events(client, rule="flaky_observer")
    assert (flaky_row["outcome"], flaky_row["effect"]) == ("errored", "observed")
    assert {f.name: f.effect for f in billed.guardrails} == {
        "card_mask_observer": "observed",
        "flaky_observer": "observed",
    }
    detail = client.get(f"/api/guardrail-events/{mask_row['event_id']}").json()
    assert detail["judged"]["recorded"] is True
    assert "4111-1111-1111-1111" in detail["judged"]["text"], "the raw tool result was judged"

    # 3 · A streamed reply a mask fires on cannot be taken back: blocked, with the reason.
    streamer = fa.Agent(
        name="streamer",
        system_prompt="Reply with exactly the word the user asks for, nothing else.",
        llm=_llm(),
        guardrails=[
            Guardrail(
                name="no_pineapple",
                guardrail_type=GuardrailType.regex,
                position=GuardrailPosition.output,
                config={"pattern": "(?i)pineapple"},
                action="mask",
            )
        ],
    )

    async def drain() -> None:
        async for _ in streamer.astream("Say the word PINEAPPLE."):
            pass

    from fastaiagent._internal.errors import GuardrailBlockedError

    with pytest.raises(GuardrailBlockedError):
        asyncio.run(drain())
    (stream_row,) = _events(client, rule="no_pineapple")
    assert (stream_row["outcome"], stream_row["effect"]) == ("blocked", "blocked")
    assert stream_row["final_reason"] and "streamed" in stream_row["final_reason"]
    assert stream_row["action_taken"] == "masked", "the wire value is unchanged"

    # 4 · Status reads the root: only the streamed (blocked) run is failing.
    from fastaiagent.trace import otel

    otel.get_tracer_provider().force_flush()
    failing = client.get("/api/traces", params={"status": "ERROR"}).json()["rows"]
    assert [r["name"] for r in failing] == ["agent.streamer"], failing
    assert client.get("/api/overview").json()["failing_traces_last_24h"] == 1
    assert not _events(client, outcome="passed", hide_passes=True)
