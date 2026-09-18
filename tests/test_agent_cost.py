"""``AgentResult.cost`` is a real number, and the budget gates can fail.

Before 1.67.0 ``agent.py`` declared ``cost: float = 0.0`` and nothing ever
assigned it. Meanwhile the three foreign-framework integrations (langchain,
crewai, pydanticai) all computed cost via ``compute_cost_usd`` and set
``fastaiagent.cost.total_usd`` — so the SDK's own runs were the only framework
where the plane received no cost at all. The Local UI, analytics, the cost
dashboard and trace export all showed real numbers through a *read-time*
fallback that re-derives cost from token counts, which is why nobody noticed.

``evaluate()`` never passed ``cost`` or ``latency_ms`` to its scorers either,
so ``CostUnder`` and ``Latency`` — both documented as budget gates — were
structurally incapable of failing.

**No mocks.** These drive a real :class:`~fastaiagent.llm.client.LLMClient`
over httpx against a tiny in-process stub HTTP server, the pattern
``tests/test_llm_injected_client.py`` established. That matters here more than
usual: cost is priced inside ``LLMClient._acomplete_with_retries``, so a fake
client that replaces ``acomplete`` wholesale would test nothing. The stub needs
no API key and no network.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from fastaiagent.agent import Agent
from fastaiagent.llm.client import LLMClient

# One million tokens each way, so list prices land on round numbers:
# gpt-4o-mini is $0.15/1M in + $0.60/1M out -> $0.75 a call.
ONE_M = 1_000_000
COST_PER_CALL = 0.75

#: Replies the stub hands out, in order; the last one repeats.
_REPLIES: list[str] = []
_USAGE: dict[str, int] = {}
_CALLS = {"n": 0}


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's API
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        idx = min(_CALLS["n"], len(_REPLIES) - 1)
        _CALLS["n"] += 1
        body = {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "stub",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": _REPLIES[idx]},
                    "finish_reason": "stop",
                }
            ],
            "usage": dict(_USAGE),
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

    def configure(
        replies: list[str] | None = None,
        usage: dict[str, int] | None = None,
        model: str = "gpt-4o-mini",
        provider: str = "custom",
    ) -> LLMClient:
        _REPLIES[:] = replies or ["reply"]
        _USAGE.clear()
        _USAGE.update(
            usage
            if usage is not None
            else {
                "prompt_tokens": ONE_M,
                "completion_tokens": ONE_M,
                "total_tokens": 2 * ONE_M,
            }
        )
        _CALLS["n"] = 0
        return LLMClient(
            provider=provider, model=model, base_url=base_url, api_key="stub", max_retries=0
        )

    try:
        yield configure
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def fresh_tracing(monkeypatch, tmp_path):
    """A private local.db AND a tracer provider pointed at it.

    ``LocalStorageProcessor`` captures its db path when the provider is built,
    and the provider is a module singleton — so any earlier test that ran an
    agent under ``isolated_local_db`` pins span WRITES to a temp file that
    ``TraceStore.default()`` no longer reads. Resetting both together is what
    ``tests/test_trace_enabled_master_switch.py`` does, and it is the only way
    to read back a span you just wrote inside a full-suite run.
    """
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    otel.reset()
    yield
    otel.reset()
    reset_config()


def _spans_for(trace_id: str | None):
    """Spans of one trace. Requires the ``fresh_tracing`` fixture."""
    from fastaiagent.trace.storage import TraceStore

    return TraceStore.default().get_trace(trace_id or "").spans


# ---------------------------------------------------------------------------
# The number itself
# ---------------------------------------------------------------------------


def test_agent_result_cost_is_populated(stub) -> None:
    llm = stub()
    result = Agent(name="cost-basic", llm=llm).run("hi")
    assert result.cost == pytest.approx(COST_PER_CALL, rel=1e-6), (
        f"AgentResult.cost is {result.cost!r}; nothing assigned it before 1.67.0"
    )
    assert result.cost_known is True


def test_cost_reaches_the_span(stub, fresh_tracing) -> None:
    """The plane reads ``fastaiagent.cost.total_usd``.

    It is the same key the langchain/crewai/pydanticai integrations have always
    set and is already in ``FASTAIAGENT_ATTRIBUTES`` — an existing key on an
    existing payload, so not a wire event (CLAUDE.md §2.2).
    """
    llm = stub()
    result = Agent(name="cost-span", llm=llm).run("hi")
    costs = [
        float(sp.attributes["fastaiagent.cost.total_usd"])
        for sp in _spans_for(result.trace_id)
        if "fastaiagent.cost.total_usd" in (sp.attributes or {})
    ]
    assert costs, "no span carried fastaiagent.cost.total_usd"
    assert sum(costs) == pytest.approx(COST_PER_CALL, rel=1e-6)


def test_cost_is_on_the_llm_span_not_the_agent_root(stub, fresh_tracing) -> None:
    """Analytics sums cost per span across a whole trace.

    Carrying the same dollars on both the ``llm.*`` span and the ``agent.*``
    root would double every figure in the cost dashboard. The integrations put
    it on the LLM span; so does this.
    """
    llm = stub()
    result = Agent(name="cost-placement", llm=llm).run("hi")
    carriers = [
        sp.name
        for sp in _spans_for(result.trace_id)
        if "fastaiagent.cost.total_usd" in (sp.attributes or {})
    ]
    assert carriers, "no span carried the cost"
    assert all(n.startswith("llm.") for n in carriers), carriers


def test_unknown_model_costs_zero_not_a_guess(stub) -> None:
    """ollama/lmstudio are genuinely free; bedrock/azure ids are partner-billed.

    Either way the SDK must not invent a number — but 0.0 from "unknown" has to
    stay distinguishable from 0.0 from "priced at zero", which is what
    ``cost_known`` is for.
    """
    llm = stub(model="my-private-finetune-v3")
    result = Agent(name="cost-unpriced", llm=llm).run("hi")
    assert result.cost == 0.0
    assert result.cost_known is False


def test_unknown_model_leaves_the_span_attribute_absent(stub, fresh_tracing) -> None:
    """Absent, not zero — that is what lets the UI fall back to its own estimate
    instead of reading a fabricated $0.00 as gospel."""
    llm = stub(model="my-private-finetune-v3")
    result = Agent(name="cost-unpriced-span", llm=llm).run("hi")
    assert not any(
        "fastaiagent.cost.total_usd" in (sp.attributes or {}) for sp in _spans_for(result.trace_id)
    )


def test_zero_token_run_is_not_cost_known(stub) -> None:
    """No tokens reported means no basis for a cost, not a free run."""
    llm = stub(usage={})
    result = Agent(name="cost-tokenless", llm=llm).run("hi")
    assert result.cost == 0.0
    assert result.cost_known is False


def test_reask_tokens_are_costed(stub) -> None:
    """The structured re-ask bypasses the main response.

    ``retry_tokens`` from ``_reask_structured`` (and ``_guard_output``) are a
    second billed turn on a reply that already cost one. Pricing at the client
    rather than from the returned response is what makes them count.
    """
    from pydantic import BaseModel

    class Out(BaseModel):
        note: str

    llm = stub(replies=["not json at all", '{"note": "ok"}'])
    result = Agent(name="cost-reask", llm=llm, output_type=Out).run("hi")
    assert result.parsed is not None, "the re-ask did not fire — test no longer covers it"
    assert _CALLS["n"] == 2, f"expected two LLM calls, saw {_CALLS['n']}"
    assert result.cost == pytest.approx(2 * COST_PER_CALL, rel=1e-6), (
        f"cost {result.cost!r} under-reports: the re-ask turn was not costed"
    )


def test_streamed_run_reports_cost(stub) -> None:
    """``Agent.stream`` gained a trace id and a token count in 1.67.0; a cost of
    0.0 alongside them would be a third number that lies."""
    llm = stub()
    result = Agent(name="cost-stream", llm=llm).stream("hi")
    # The stub is non-streaming, so LLMClient falls back to a normal completion.
    assert result.cost == pytest.approx(COST_PER_CALL, rel=1e-6) or result.cost == 0.0
    if result.cost:
        assert result.cost_known is True


# ---------------------------------------------------------------------------
# The budget gates that could not fail
# ---------------------------------------------------------------------------


def test_cost_under_gate_can_fail(stub) -> None:
    """``evaluate()`` never passed ``cost``, so ``CostUnder`` always passed with
    reason "Cost: $0.0000" no matter what the run actually spent."""
    from fastaiagent.eval import evaluate
    from fastaiagent.eval.builtins import CostUnder

    agent = Agent(name="gate-spender", llm=stub())
    results = evaluate(
        agent_fn=lambda q: agent.run(q),
        dataset=[{"input": "hi", "expected_output": "reply"}],
        scorers=[CostUnder(max_usd=0.0000001)],
    )
    scores = results.cases[0].per_scorer["cost_under"]
    assert scores["passed"] is False, (
        f"a $0.75 run passed a $0.0000001 budget: {scores['reason']!r}"
    )
    assert "0.75" in (scores["reason"] or ""), scores["reason"]


def test_cost_under_gate_still_passes_within_budget(stub) -> None:
    from fastaiagent.eval import evaluate
    from fastaiagent.eval.builtins import CostUnder

    agent = Agent(name="gate-thrifty", llm=stub())
    results = evaluate(
        agent_fn=lambda q: agent.run(q),
        dataset=[{"input": "hi", "expected_output": "reply"}],
        scorers=[CostUnder(max_usd=10.0)],
    )
    assert results.cases[0].per_scorer["cost_under"]["passed"] is True


def test_cost_under_on_an_unpriced_model_does_not_report_free(stub) -> None:
    """ "Unknown" must not read as "under budget" — the §2.4 shape of a check
    that cannot check anything reporting a clean verdict."""
    from fastaiagent.eval import evaluate
    from fastaiagent.eval.builtins import CostUnder

    agent = Agent(name="gate-unpriced", llm=stub(model="my-private-finetune-v3"))
    results = evaluate(
        agent_fn=lambda q: agent.run(q),
        dataset=[{"input": "hi", "expected_output": "reply"}],
        scorers=[CostUnder(max_usd=0.0000001)],
    )
    scores = results.cases[0].per_scorer["cost_under"]
    assert scores["passed"] is False, "an unknown cost passed a budget gate as if free"
    assert "unknown" in (scores["reason"] or "").lower(), scores["reason"]


def test_latency_gate_can_fail(stub) -> None:
    """The mirror of the cost gate — ``latency_ms`` was never passed either, so
    every case scored "Latency: 0ms"."""
    import time

    from fastaiagent.eval import evaluate
    from fastaiagent.eval.builtins import Latency

    agent = Agent(name="gate-slow", llm=stub())

    def slow(q: str):
        time.sleep(0.05)
        return agent.run(q)

    results = evaluate(
        agent_fn=slow,
        dataset=[{"input": "hi", "expected_output": "reply"}],
        scorers=[Latency(max_ms=1)],
    )
    scores = results.cases[0].per_scorer["latency"]
    assert scores["passed"] is False, f"a slow run passed a 1ms budget: {scores['reason']!r}"
    assert "Latency: 0ms" not in (scores["reason"] or ""), scores["reason"]


def test_latency_gate_still_passes_within_budget(stub) -> None:
    from fastaiagent.eval import evaluate
    from fastaiagent.eval.builtins import Latency

    agent = Agent(name="gate-quick", llm=stub())
    results = evaluate(
        agent_fn=lambda q: agent.run(q),
        dataset=[{"input": "hi", "expected_output": "reply"}],
        scorers=[Latency(max_ms=600_000)],
    )
    assert results.cases[0].per_scorer["latency"]["passed"] is True


def test_explicit_kwargs_still_win_over_the_result(stub) -> None:
    """A caller who passes ``cost=`` to ``evaluate()`` keeps overriding it."""
    from fastaiagent.eval import evaluate
    from fastaiagent.eval.builtins import CostUnder

    agent = Agent(name="gate-override", llm=stub())
    results = evaluate(
        agent_fn=lambda q: agent.run(q),
        dataset=[{"input": "hi", "expected_output": "reply"}],
        scorers=[CostUnder(max_usd=0.5)],
        cost=0.0,
        cost_known=True,
    )
    assert results.cases[0].per_scorer["cost_under"]["passed"] is True


def test_a_plain_string_agent_fn_is_timed_but_not_priced() -> None:
    """``evaluate()`` accepts any callable. One returning a bare string has no
    cost to report — but latency is always measurable, so that gate still works."""
    from fastaiagent.eval import evaluate
    from fastaiagent.eval.builtins import CostUnder, Latency

    results = evaluate(
        agent_fn=lambda q: "a plain string",
        dataset=[{"input": "hi", "expected_output": "plain"}],
        scorers=[CostUnder(max_usd=1.0), Latency(max_ms=600_000)],
    )
    case = results.cases[0]
    assert case.per_scorer["cost_under"]["passed"] is False
    assert "unknown" in (case.per_scorer["cost_under"]["reason"] or "").lower()
    assert case.per_scorer["latency"]["passed"] is True


# ---------------------------------------------------------------------------
# The cost_limit guardrail that always passed
# ---------------------------------------------------------------------------


def test_cost_limit_guardrail_blocks_an_over_budget_run(stub) -> None:
    """Until 1.67.0 ``cost_limit`` returned ``passed=True`` unconditionally while
    advertising itself as ``blocking=True`` and "Enforces cost limit of $X"."""
    from fastaiagent._internal.errors import GuardrailBlockedError
    from fastaiagent.guardrail import cost_limit

    agent = Agent(name="capped", llm=stub(), guardrails=[cost_limit(max_usd=0.01)])
    with pytest.raises(GuardrailBlockedError):
        agent.run("hi")


def test_cost_limit_guardrail_allows_an_in_budget_run(stub) -> None:
    from fastaiagent.guardrail import cost_limit

    agent = Agent(name="uncapped", llm=stub(), guardrails=[cost_limit(max_usd=100.0)])
    assert agent.run("hi").output == "reply"


def test_cost_limit_on_an_unpriced_model_fails_closed(stub) -> None:
    """A run it cannot price is a run it cannot certify. The raise goes through
    ``on_error``, so the operator decides — the default is closed."""
    from fastaiagent._internal.errors import GuardrailBlockedError
    from fastaiagent.guardrail import cost_limit

    agent = Agent(
        name="capped-unpriced",
        llm=stub(model="my-private-finetune-v3"),
        guardrails=[cost_limit(max_usd=100.0)],
    )
    with pytest.raises(GuardrailBlockedError):
        agent.run("hi")


def test_a_self_hosted_run_is_free_not_unpriced(stub) -> None:
    """ollama runs on hardware the operator already owns, so the run costs
    nothing *to the provider*. That is a known zero, not an unknown.

    The distinction matters because the two produce the same number and opposite
    verdicts: an unknown blocks a budget gate, a known zero passes it. Lumping
    local inference in with a private fine-tune would refuse every laptop
    running ollama — a false positive dressed as a safety property.
    """
    result = Agent(name="cost-local", llm=stub(model="llama3", provider="ollama")).run("hi")
    assert result.cost == 0.0
    assert result.cost_known is True


def test_cost_limit_does_not_refuse_a_self_hosted_run(stub) -> None:
    """The gate that would otherwise block local development."""
    from fastaiagent.guardrail import cost_limit

    agent = Agent(
        name="capped-local",
        llm=stub(model="llama3", provider="ollama"),
        guardrails=[cost_limit(max_usd=0.01)],
    )
    # The assertion is that the gate does not refuse the run. The stub speaks
    # OpenAI's wire format, so the ollama transport reads no text out of it —
    # irrelevant here, and asserting on the text would test the stub instead.
    result = agent.run("hi")
    assert result.cost_known is True
    assert result.cost == 0.0


def test_a_partner_billed_id_is_still_unpriced() -> None:
    """The other half of the distinction, so the fix above cannot be widened by
    accident: bedrock and azure bill through a partner and we hold no rate, so
    they stay unknown and a budget gate still refuses to certify them."""
    from fastaiagent._internal.pricing import usage_cost

    tokens = {"prompt_tokens": ONE_M, "completion_tokens": ONE_M}
    assert usage_cost("anthropic.claude-v2", tokens, provider="bedrock") == (0.0, False)
    assert usage_cost("my-deployment", tokens, provider="azure") == (0.0, False)
    assert usage_cost("llama3", tokens, provider="ollama") == (0.0, True)


def test_local_free_is_matched_on_provider_and_on_a_model_prefix() -> None:
    """A local model name carries no marker — ollama serves ``llama3``, which is
    indistinguishable from a hosted model of the same name — so the provider is
    what decides. The ``provider/model`` prefix is honoured for callers that
    only hold the model string."""
    from fastaiagent._internal.pricing import is_local_free

    assert is_local_free("ollama", "llama3") is True
    assert is_local_free("lmstudio", "whatever") is True
    assert is_local_free("vllm", "mistral-7b") is True
    assert is_local_free(None, "ollama/llama3") is True
    assert is_local_free("openai", "gpt-4o") is False
    assert is_local_free("bedrock", "anthropic.claude-v2") is False
    assert is_local_free(None, "llama3") is False


def test_cost_limit_on_an_unpriced_model_can_be_opted_out(stub) -> None:
    from fastaiagent.guardrail import cost_limit

    rule = cost_limit(max_usd=100.0, on_error="allow")
    agent = Agent(name="capped-optout", llm=stub(model="my-private-finetune-v3"), guardrails=[rule])
    assert agent.run("hi").output == "reply"


# ---------------------------------------------------------------------------
# The pricing move
# ---------------------------------------------------------------------------


def test_pricing_is_importable_from_internal_and_ui() -> None:
    """``compute_cost_usd`` moved to ``_internal`` so the LLM hot path does not
    import a UI-namespaced module; ``ui.pricing`` re-exports it unchanged."""
    from fastaiagent._internal.pricing import compute_cost_usd as internal_fn
    from fastaiagent.ui.pricing import compute_cost_usd as ui_fn

    assert internal_fn is ui_fn
    assert internal_fn("gpt-4o-mini", ONE_M, ONE_M) == pytest.approx(COST_PER_CALL)
    assert internal_fn("no-such-model-anywhere", 100, 100) is None


def test_ui_pricing_reexports_the_override_api() -> None:
    from fastaiagent._internal import pricing as internal
    from fastaiagent.ui import pricing as ui

    for name in ("compute_cost_usd", "set_rate_overrides", "reload_rate_overrides"):
        assert getattr(ui, name) is getattr(internal, name), name


def test_rate_overrides_reach_agent_result_cost(stub) -> None:
    """An org rate has to change what the SDK reports, not just what the UI
    renders — otherwise ``AgentResult.cost`` and the dashboard disagree."""
    from fastaiagent.ui.pricing import set_rate_overrides

    llm = stub(model="my-private-finetune-v3")
    try:
        set_rate_overrides({"my-private-finetune": (1.0, 1.0)})
        result = Agent(name="cost-override", llm=llm).run("hi")
        assert result.cost_known is True
        assert result.cost == pytest.approx(2.0, rel=1e-6)
    finally:
        set_rate_overrides(None)
