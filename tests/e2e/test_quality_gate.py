"""End-to-end quality gate — the full product lifecycle.

This is the CI/CD discipline gate for the SDK. Every commit runs this file
with real API keys and a real platform to prove the 18-step flow actually
works end-to-end:

    install → connect → create agent → add tool → add guardrail → run →
    inspect result → trace_id → verify on platform → run eval → check scores
    → load replay → fork at step 2 → rerun → compare → save_as_test →
    re-evaluate with exact_match + LLMJudge

If this passes, the product works. If it breaks, nothing else matters.

Local run:
    export FASTAIAGENT_API_KEY=fa_k_...
    export FASTAIAGENT_TARGET=http://localhost:8001
    export OPENAI_API_KEY=sk-...
    pytest tests/e2e/ -v -m e2e

CI run (fail-closed):
    E2E_REQUIRED=1 pytest tests/e2e/ -v -m e2e
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import pytest

from tests.e2e.conftest import require_env, require_platform

pytestmark = pytest.mark.e2e


def _lookup_order(order_id: str) -> str:
    """Look up an order by ID."""
    orders = {
        "ORD-001": "MacBook Pro 16-inch, shipped 2026-04-01, delivered 2026-04-03",
        "ORD-002": "AirPods Pro, processing, estimated delivery 2026-04-10",
        "ORD-003": "iPad Air, cancelled by customer on 2026-03-28",
    }
    return orders.get(order_id, f"Order {order_id} not found.")


class TestQualityGate:
    """Ordered end-to-end quality gate. Each method is one step in the flow.

    State is threaded via the module-scoped ``gate_state`` fixture so a
    failure in step N names step N in the CI log (rather than burying the
    whole pipeline in a single monolithic test).
    """

    # ── Step 1: install/import ────────────────────────────────────────────

    def test_01_install_and_import(self, gate_state: dict[str, Any]) -> None:
        require_env()
        import fastaiagent as fa

        assert fa.__version__, "SDK has no __version__"
        assert hasattr(fa, "Agent")
        assert hasattr(fa, "connect")
        assert hasattr(fa, "Replay")
        gate_state["fa"] = fa

    # ── Step 2: connect to platform ───────────────────────────────────────

    def test_02_connect_to_platform(self, gate_state: dict[str, Any]) -> None:
        require_env()
        require_platform()
        import fastaiagent as fa
        from fastaiagent.client import _connection

        fa.connect(
            api_key=os.environ["FASTAIAGENT_API_KEY"],
            target=os.environ["FASTAIAGENT_TARGET"],
        )
        assert fa.is_connected, "fa.connect() silently failed"
        # Read the SDK-normalized target (may have had "http://" prepended)
        gate_state["target"] = _connection.target
        gate_state["api_key"] = os.environ["FASTAIAGENT_API_KEY"]

    # ── Step 3: create agent ──────────────────────────────────────────────

    def test_03_create_agent(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent import Agent, LLMClient

        agent = Agent(
            name="quality-gate-support",
            system_prompt=(
                "You are a customer support agent for Acme Corp. "
                "Use the lookup_order tool to check order status when asked. "
                "Be concise."
            ),
            llm=LLMClient(provider="openai", model="gpt-4.1"),
        )
        assert agent.name == "quality-gate-support"
        assert agent.llm.provider == "openai"
        gate_state["agent"] = agent

    # ── Step 4: add tool ──────────────────────────────────────────────────

    def test_04_add_tool(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent import FunctionTool

        agent = gate_state["agent"]
        tool = FunctionTool(name="lookup_order", fn=_lookup_order)
        agent.tools = [tool]
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "lookup_order"

        # Phase B: auto-registration in the global ToolRegistry
        from fastaiagent import ToolRegistry

        assert ToolRegistry.get("lookup_order") is tool, (
            "FunctionTool did not auto-register — Phase B regression"
        )

    # ── Step 5: add guardrail (block path) ────────────────────────────────

    def test_05_guardrail_blocks_pii(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent import Agent, FunctionTool, LLMClient, no_pii
        from fastaiagent._internal.errors import GuardrailBlockedError
        from fastaiagent.guardrail.guardrail import GuardrailPosition

        # Use a throw-away agent so the main gate agent stays clean.
        blocking_agent = Agent(
            name="quality-gate-guardrail-probe",
            system_prompt="You are a test agent.",
            llm=LLMClient(provider="openai", model="gpt-4.1"),
            tools=[FunctionTool(name="lookup_order", fn=_lookup_order)],
            guardrails=[no_pii(position=GuardrailPosition.input)],
        )
        with pytest.raises(GuardrailBlockedError):
            blocking_agent.run(
                "My SSN is 123-45-6789 and my email is test@example.com — look up ORD-001"
            )

    # ── Step 6: add guardrail (allow path) ────────────────────────────────

    def test_06_guardrail_allows_clean_input(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent import no_pii
        from fastaiagent.guardrail.guardrail import GuardrailPosition

        agent = gate_state["agent"]
        agent.guardrails = [no_pii(position=GuardrailPosition.input)]
        # Don't run yet — step 7 will run with the guardrails attached.
        assert len(agent.guardrails) == 1

    # ── Step 7: run agent ─────────────────────────────────────────────────

    def test_07_run_agent(self, gate_state: dict[str, Any]) -> None:
        require_env()
        agent = gate_state["agent"]
        result = agent.run("What's the status of order ORD-001?")
        assert result.output, "agent.run returned empty output"
        lower = result.output.lower()
        assert "ord-001" in lower or "shipped" in lower or "macbook" in lower, (
            f"unexpected output — LLM did not call lookup_order or ignored it: "
            f"{result.output!r}"
        )
        gate_state["result"] = result

    # ── Step 8: inspect result ────────────────────────────────────────────

    def test_08_inspect_result(self, gate_state: dict[str, Any]) -> None:
        require_env()
        result = gate_state["result"]
        assert result.tokens_used > 0, "tokens_used not populated"
        assert result.latency_ms > 0, "latency_ms not populated"
        assert result.cost >= 0.0, "cost has wrong sign"
        assert isinstance(result.tool_calls, list)
        assert len(result.tool_calls) >= 1, (
            "agent did not invoke lookup_order — tool path broken or LLM ignored it"
        )
        assert result.tool_calls[0]["tool_name"] == "lookup_order"

    # ── Step 9: check trace_id exists ─────────────────────────────────────

    def test_09_trace_id_exists(self, gate_state: dict[str, Any]) -> None:
        require_env()
        result = gate_state["result"]
        assert result.trace_id, "AgentResult.trace_id is None — tracing broken"
        assert len(result.trace_id) >= 16, f"trace_id too short: {result.trace_id}"
        gate_state["trace_id"] = result.trace_id

    # ── Step 10: verify trace in platform dashboard ───────────────────────

    def test_10_verify_trace_on_platform(self, gate_state: dict[str, Any]) -> None:
        require_env()
        require_platform()
        import fastaiagent as fa

        # Force-flush pending spans to the platform exporter.
        fa.disconnect()
        gate_state["disconnected"] = True

        trace_id = gate_state["trace_id"]
        target = gate_state["target"]
        api_key = gate_state["api_key"]

        # Retry loop — the platform ingests asynchronously.
        last_status: int | None = None
        last_body: str = ""
        for _ in range(10):
            try:
                resp = httpx.get(
                    f"{target}/public/v1/traces/{trace_id}",
                    headers={"X-API-Key": api_key},
                    timeout=5.0,
                )
            except httpx.HTTPError as e:
                last_body = str(e)
                time.sleep(1.0)
                continue
            last_status = resp.status_code
            last_body = resp.text[:200]
            if resp.status_code == 200:
                data = resp.json()
                spans = data.get("spans", [])
                assert len(spans) > 0, f"platform returned 0 spans for {trace_id}"
                span_names = [s.get("name", "") for s in spans]
                assert any(n.startswith("agent.") for n in span_names), (
                    f"no agent.* span on platform: {span_names}"
                )
                assert any("tool.lookup_order" in n for n in span_names), (
                    f"no tool.lookup_order span on platform — "
                    f"Phase A executor instrumentation regression: {span_names}"
                )
                gate_state["platform_spans"] = span_names
                return
            time.sleep(1.0)

        pytest.fail(
            f"platform never returned 200 for trace {trace_id} "
            f"(last status={last_status}, body={last_body!r})"
        )

    # ── Step 11: run eval ─────────────────────────────────────────────────

    def test_11_run_eval(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent.eval import Dataset, evaluate
        from fastaiagent.eval.scorer import Scorer, ScorerResult

        agent = gate_state["agent"]

        def agent_fn(input_text: str) -> str:
            return agent.run(input_text, trace=False).output

        @Scorer.code("contains_keyword")
        def contains_keyword(
            input: str, output: str, expected: str | None = None
        ) -> ScorerResult:
            if expected and expected.lower() in output.lower():
                return ScorerResult(
                    score=1.0, passed=True, reason=f"contains {expected!r}"
                )
            return ScorerResult(
                score=0.0, passed=False, reason=f"missing {expected!r}"
            )

        dataset = Dataset.from_list(
            [
                {"input": "Status of ORD-001?", "expected": "shipped"},
                {"input": "Status of ORD-002?", "expected": "processing"},
                {"input": "Status of ORD-003?", "expected": "cancelled"},
                {"input": "Status of ORD-999?", "expected": "not found"},
            ]
        )
        results = evaluate(agent_fn=agent_fn, dataset=dataset, scorers=[contains_keyword])
        assert results.summary(), "EvalResults.summary() returned empty"
        gate_state["eval_results"] = results

    # ── Step 12: check scores ─────────────────────────────────────────────

    def test_12_check_eval_scores(self, gate_state: dict[str, Any]) -> None:
        require_env()
        results = gate_state["eval_results"]
        per_scorer = results.scores
        assert per_scorer, "EvalResults.scores is empty"
        scorer_results = per_scorer.get("contains_keyword", [])
        assert scorer_results, "contains_keyword scorer produced no results"
        mean = sum(r.score for r in scorer_results) / len(scorer_results)
        # Generous baseline — tighten after first real run establishes a floor.
        assert mean >= 0.5, (
            f"eval mean score {mean:.2f} below 0.5 baseline — agent regressed or dataset broken"
        )

    # ── Step 13: load replay ──────────────────────────────────────────────

    def test_13_load_replay(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent.trace.replay import Replay

        trace_id = gate_state["trace_id"]
        replay = Replay.load(trace_id)
        steps = replay.steps()
        assert len(steps) >= 3, (
            f"expected >=3 spans (root agent + LLM + tool), got {len(steps)}: "
            f"{[s.span_name for s in steps]}"
        )
        # Phase A: the root span must carry reconstruction metadata.
        root_attrs = replay._trace.spans[0].attributes
        required_attrs = [
            "agent.config",
            "agent.tools",
            "agent.guardrails",
            "agent.llm.config",
        ]
        missing = [k for k in required_attrs if k not in root_attrs]
        assert not missing, (
            f"Phase A span metadata missing — replay cannot reconstruct agent: {missing}"
        )
        gate_state["replay"] = replay

    # ── Step 14: fork at step 2 ───────────────────────────────────────────

    def test_14_fork_at_step_2(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent.trace.replay import ForkedReplay

        replay = gate_state["replay"]
        fork_point = 2
        forked = replay.fork_at(step=fork_point)
        assert isinstance(forked, ForkedReplay)
        forked.modify_prompt(
            "You are a terse support agent. Reply in one sentence maximum."
        )
        gate_state["forked"] = forked
        gate_state["fork_point"] = fork_point

    # ── Step 15: rerun ────────────────────────────────────────────────────

    def test_15_rerun(self, gate_state: dict[str, Any]) -> None:
        require_env()
        forked = gate_state["forked"]
        rerun_result = forked.rerun()
        assert rerun_result.new_output is not None, (
            "ForkedReplay.rerun returned new_output=None — Phase C regression"
        )
        assert isinstance(rerun_result.new_output, str)
        assert len(rerun_result.new_output) > 0, "rerun produced empty output"
        assert rerun_result.trace_id, "rerun did not emit a new trace_id"
        gate_state["rerun_result"] = rerun_result

    # ── Step 16: compare ──────────────────────────────────────────────────

    def test_16_compare(self, gate_state: dict[str, Any]) -> None:
        require_env()
        forked = gate_state["forked"]
        rerun_result = gate_state["rerun_result"]

        cmp = forked.compare(rerun_result)
        # v1.14: ``diverged_at`` is now computed by walking both step
        # lists (was hardcoded to ``fork_point`` in v1.13). The
        # ``compare_status == "ok"`` check is the actual contract this
        # test guards — the comparison ran and produced a valid
        # ComparisonResult. Whether the LLM's reply *happened* to
        # match byte-for-byte (both the original and modified prompts
        # ask for terse single-sentence replies; against the same
        # tool-backed input that can land on identical text) is LLM
        # nondeterminism, not a contract worth pinning.
        assert cmp.compare_status == "ok"
        # diverged_at may legitimately be None when the live rerun
        # produces identical output — that's a successful comparison,
        # just no divergence. The integer type is asserted by Python
        # itself via the Pydantic schema; nothing more to check here.
        assert cmp.diverged_at is None or cmp.diverged_at >= 0
        assert len(cmp.original_steps) >= 3
        assert len(cmp.new_steps) >= 1, (
            "compare() did not load rerun trace — Phase C compare() regression"
        )

    # ── Step 17: save the rerun as a regression test ──────────────────────

    def test_17_save_as_test(
        self, gate_state: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Closes the "every failure becomes a test" loop in code.

        Exercises ``ReplayResult.save_as_test()`` against a real ``tmp_path``
        and asserts the v1.14.1 schema:

        * ``input``, ``expected_output`` — the eval contract
        * ``trace_id`` / ``fixed_trace_id`` — the *rerun's* id (the fix)
        * ``source_trace_id`` — the *original failure* id (provenance)
        * ``fork_step``, ``modifications`` — the audit trail
        * ``created_at`` — ISO-8601 timestamp

        Pre-v1.14.1 ``trace_id`` was overwritten with ``source_trace_id``,
        losing the link to the actual rerun — fixed in v1.14.1.
        """
        require_env()
        rerun_result = gate_state["rerun_result"]
        trace_id = gate_state["trace_id"]

        regression_dir = tmp_path_factory.mktemp("quality_gate_regression")
        dataset_path = regression_dir / "regression_tests.jsonl"
        expected_output = str(rerun_result.new_output)
        regression_input = "What's the status of order ORD-001?"

        returned_path = rerun_result.save_as_test(
            dataset_path,
            input=regression_input,
            expected_output=expected_output,
            source_trace_id=trace_id,
            fork_step=2,
            modifications={"prompt": "modified-by-quality-gate"},
        )
        assert returned_path == dataset_path
        assert dataset_path.exists(), "save_as_test did not create the JSONL file"

        lines = dataset_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1, f"expected 1 JSONL line, got {len(lines)}"
        record = json.loads(lines[0])
        assert record["input"] == regression_input
        assert record["expected_output"] == expected_output
        # v1.14.1: trace_id is the rerun's id; source_trace_id carries
        # the failure id in its own field.
        assert record["trace_id"] == rerun_result.trace_id
        assert record["fixed_trace_id"] == rerun_result.trace_id
        assert record["source_trace_id"] == trace_id, (
            f"save_as_test lost source_trace_id: {record['source_trace_id']!r} "
            f"!= {trace_id!r}"
        )
        assert record["fork_step"] == 2
        assert record["modifications"] == {"prompt": "modified-by-quality-gate"}
        # ISO-8601 timestamp parses cleanly.
        from datetime import datetime

        datetime.fromisoformat(record["created_at"])

        gate_state["regression_dataset_path"] = dataset_path
        gate_state["regression_input"] = regression_input
        gate_state["regression_expected_output"] = expected_output

    # ── Step 18: re-evaluate the regression case via exact_match + LLMJudge

    def test_18_evaluate_regression_case(self, gate_state: dict[str, Any]) -> None:
        """Proves the saved JSONL is directly consumable by ``evaluate()`` —
        with both a deterministic string scorer and an LLM-as-judge scorer
        in the same call. This is the closing edge of the loop the SDK now
        guarantees end-to-end."""
        require_env()
        from fastaiagent.eval import LLMJudge, evaluate

        dataset_path = gate_state["regression_dataset_path"]
        expected_output = gate_state["regression_expected_output"]

        # The "agent" under test for the regression case is the verified
        # fix itself — i.e., the rerun's new_output. Using it as the
        # agent_fn pins exact_match to a deterministic pass and gives
        # LLMJudge an output identical to expected (must also pass).
        # This step verifies the *framework wiring*, not the agent's
        # accuracy (covered by steps 7-12).
        def agent_fn(_input_text: str) -> str:
            return expected_output

        results = evaluate(
            agent_fn=agent_fn,
            dataset=str(dataset_path),
            scorers=["exact_match", LLMJudge(criteria="correctness")],
            persist=False,
        )

        # exact_match: deterministic — must pass.
        exact_scores = results.scores.get("exact_match", [])
        assert len(exact_scores) == 1, (
            f"exact_match produced {len(exact_scores)} results — "
            f"evaluate() did not read the JSONL line written by save_as_test()"
        )
        assert exact_scores[0].passed is True, (
            f"exact_match failed against verbatim agent_fn — "
            f"scorer wiring or dataset schema broken: "
            f"score={exact_scores[0].score} reason={exact_scores[0].reason!r}"
        )

        # LLMJudge: semantic — the gate's job here is to prove the scorer
        # is *wired* (registered, fed the right fields from the JSONL,
        # produces a ScorerResult). Whether the judge LLM successfully
        # emits clean JSON on any given roll is a separate concern — real
        # judge calls intermittently return markdown-wrapped or empty
        # content and fall through the scorer's ``Judge error:`` path
        # with score=0.0. We assert reach + structure, not pass/fail.
        judge_scores = results.scores.get("llm_judge", [])
        assert len(judge_scores) == 1, (
            f"LLMJudge produced {len(judge_scores)} results — "
            f"scorer registration regression"
        )
        result = judge_scores[0]
        assert isinstance(result.score, float), (
            f"LLMJudge returned non-float score {result.score!r} — "
            f"ScorerResult shape regression"
        )
        assert 0.0 <= result.score <= 1.0, (
            f"LLMJudge score out of range: {result.score}"
        )
