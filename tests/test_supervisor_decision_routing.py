"""``Supervisor(routing="decisions")`` — a decision-model supervisor (1.84.0).

Swept across the axes a Supervisor promises (CLAUDE.md §3): ``run`` / ``arun`` /
``astream`` / ``stream`` / ``resume`` × tracing × validation on/off × fallback.

No mocks: routing and review go to the local ``/v1/decisions`` stand-in through
the real ``LLMClient``; workers are real ``Agent``s on ``FunctionModel``.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from fastaiagent.agent import Agent, Supervisor, SupervisorRoute, Worker
from fastaiagent.chain.interrupt import Resume, interrupt
from fastaiagent.checkpointers import SQLiteCheckpointer
from fastaiagent.llm import Predicate, Score
from fastaiagent.llm.message import MessageRole
from fastaiagent.llm.stream import HandoffEvent, TextDelta
from fastaiagent.testing import FunctionModel, TestModel
from fastaiagent.tool import FunctionTool
from fastaiagent.trace import otel


def _worker(role: str, reply: str, seen: list | None = None) -> Worker:
    def fn(messages):
        if seen is not None:
            seen.append(messages)
        return reply

    return Worker(
        agent=Agent(name=f"w-{role}", llm=FunctionModel(fn)),
        role=role,
        description=f"Handles {role} requests.",
    )


def _team(stub, *, seen=None, **kw) -> Supervisor:
    return Supervisor(
        name="desk",
        # The chat llm is unused in decisions routing; a TestModel proves no call.
        llm=TestModel(response="SHOULD NOT BE CALLED"),
        workers=[
            _worker("complaint", "complaint handled", seen),
            _worker("product_enquiry", "product answered", seen),
            _worker("other", "general help", seen),
        ],
        routing="decisions",
        router_llm=stub.client(),
        fallback_worker="other",
        **kw,
    )


def _decisions(stub) -> list[dict]:
    return [r for r in stub.requests if r["path"].endswith("/decisions")]


class TestRouting:
    def test_routes_to_the_chosen_worker_and_returns_its_reply(self, decisions_stub) -> None:
        decisions_stub.choices["worker"] = "product_enquiry"
        result = _team(decisions_stub).run("Does the lamp work with USB-C?")
        assert result.output == "product answered"  # no synthesis turn
        route = result.route
        assert isinstance(route, SupervisorRoute)
        assert (route.worker, route.chosen, route.fallback) == ("product_enquiry", "product_enquiry", False)
        assert route.confidence == 1.0
        # The Choice is built from the workers' roles and descriptions.
        q = _decisions(decisions_stub)[0]["body"]["questions"][0]
        assert q["name"] == "worker"
        assert [c["value"] for c in q["choices"]] == ["complaint", "product_enquiry", "other"]
        assert q["choices"][0]["description"] == "Handles complaint requests."

    def test_arun(self, decisions_stub) -> None:
        decisions_stub.choices["worker"] = "complaint"
        result = asyncio.run(_team(decisions_stub).arun("My order is broken"))
        assert result.output == "complaint handled"

    def test_a_refusal_goes_to_the_fallback_worker(self, decisions_stub) -> None:
        decisions_stub.refuse.add("worker")
        result = _team(decisions_stub).run("???")
        assert result.output == "general help"
        assert result.route.fallback and result.route.chosen is None
        assert result.route.confidence is None

    def test_low_confidence_goes_to_the_fallback_worker(self, decisions_stub) -> None:
        decisions_stub.script = [
            (
                200,
                {
                    "model": "gpt-6-luna",
                    "answers": [
                        {
                            "type": "choice",
                            "name": "worker",
                            "choice": "complaint",
                            "probabilities": [
                                {"value": "complaint", "probability": 0.4},
                                {"value": "product_enquiry", "probability": 0.35},
                                {"value": "other", "probability": 0.25},
                            ],
                            "confidence": 0.2,
                        }
                    ],
                    "usage": {"input_tokens": 80, "output_tokens": 0, "total_tokens": 80},
                },
            )
        ]
        result = _team(decisions_stub, routing_min_confidence=0.6).run("hmm")
        assert result.output == "general help"
        assert result.route.fallback and result.route.chosen == "complaint"

    def test_routing_questions_are_asked_in_the_same_call_and_noted_for_the_worker(
        self, decisions_stub
    ) -> None:
        decisions_stub.choices["worker"] = "complaint"
        decisions_stub.probabilities["urgent"] = 0.97
        seen: list = []
        team = _team(
            decisions_stub,
            seen=seen,
            routing_questions={
                "urgent": Predicate(instructions="The customer needs an answer today."),
                "mood": Score(instructions="How upset?", levels=["Calm", "Angry"]),
            },
        )
        result = team.run("Broken again, fix it today!")
        assert len(_decisions(decisions_stub)) == 1  # one call, three questions
        assert [q["name"] for q in _decisions(decisions_stub)[0]["body"]["questions"]] == [
            "worker",
            "urgent",
            "mood",
        ]
        assert result.route.answers.predicates["urgent"].probability == 0.97
        user_text = next(m.content for m in seen[0] if m.role == MessageRole.user)
        assert "Broken again, fix it today!" in user_text
        assert "[Supervisor routing note] urgent: yes (p=0.97); mood: Angry" in user_text

    def test_cost_and_tokens_include_the_routing_call(self, decisions_stub) -> None:
        result = _team(decisions_stub).run("hi")
        assert result.route.cost_usd == pytest.approx(100 * 0.10 / 1e6)
        assert result.cost >= result.route.cost_usd
        assert result.tokens_used >= 100


class TestReview:
    def test_reject_then_accept(self, decisions_stub) -> None:
        def reply(p):
            return 200, {
                "model": "gpt-6-luna",
                "answers": [{"type": "predicate", "name": "approved", "probability": p}],
                "usage": {"input_tokens": 50, "output_tokens": 0},
            }

        route_reply = (
            200,
            {
                "model": "gpt-6-luna",
                "answers": [
                    {
                        "type": "choice",
                        "name": "worker",
                        "choice": "complaint",
                        "probabilities": [{"value": "complaint", "probability": 1.0}],
                        "confidence": 1.0,
                    }
                ],
                "usage": {"input_tokens": 50, "output_tokens": 0},
            },
        )
        decisions_stub.script = [route_reply, reply(0.2), reply(0.9)]
        seen: list = []
        team = _team(
            decisions_stub,
            seen=seen,
            validate_outputs=True,
            validation_mode="decisions",
            validation_llm=decisions_stub.client(),
        )
        result = team.run("refund please")
        assert result.route.reviews == [0.2, 0.9]
        assert len(seen) == 2  # the worker ran twice
        retry_text = next(m.content for m in seen[1] if m.role == MessageRole.user)
        assert "[Supervisor feedback]" in retry_text and "p=0.20" in retry_text

    def test_custom_validation_criteria_is_the_question(self, decisions_stub) -> None:
        decisions_stub.probabilities["approved"] = 0.9
        team = _team(
            decisions_stub,
            validate_outputs=True,
            validation_mode="decisions",
            validation_llm=decisions_stub.client(),
            validation_criteria="The reply tells the customer the next step.",
        )
        team.run("x")
        review = _decisions(decisions_stub)[-1]["body"]["questions"][0]
        assert review == {
            "type": "predicate",
            "instructions": "The reply tells the customer the next step.",
            "name": "approved",
        }

    def test_blank_validation_criteria_is_refused(self) -> None:
        with pytest.raises(ValueError, match="validation_criteria"):
            Supervisor(name="s", llm=TestModel(), validation_criteria="  ")


class TestStreaming:
    def test_astream_hands_off_then_streams_the_worker(self, decisions_stub) -> None:
        decisions_stub.choices["worker"] = "other"

        async def collect():
            return [e async for e in _team(decisions_stub).astream("opening hours?")]

        events = asyncio.run(collect())
        assert isinstance(events[0], HandoffEvent)
        assert (events[0].from_agent, events[0].to_agent) == ("desk", "other")
        assert "confidence 1.00" in events[0].reason
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "general help"

    def test_astream_with_review_emits_only_the_reviewed_reply(self, decisions_stub) -> None:
        decisions_stub.choices["worker"] = "other"
        decisions_stub.probabilities["approved"] = 0.95
        team = _team(
            decisions_stub,
            validate_outputs=True,
            validation_mode="decisions",
            validation_llm=decisions_stub.client(),
        )

        async def collect():
            return [e async for e in team.astream("x")]

        texts = [e.text for e in asyncio.run(collect()) if isinstance(e, TextDelta)]
        assert texts == ["general help"]

    def test_stream_sync(self, decisions_stub) -> None:
        decisions_stub.choices["worker"] = "complaint"
        assert _team(decisions_stub).stream("broken").output == "complaint handled"


class TestTracing:
    @pytest.fixture
    def fresh_tracing(self, monkeypatch, tmp_path):
        from fastaiagent._internal.config import reset_config

        monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
        reset_config()
        otel.reset()
        yield tmp_path / "local.db"
        otel.reset()
        reset_config()

    def test_one_trace_route_and_worker_nested(self, decisions_stub, fresh_tracing) -> None:
        from fastaiagent.trace.storage import TraceStore

        decisions_stub.choices["worker"] = "product_enquiry"
        result = _team(decisions_stub).run("lamp?")
        otel.get_tracer_provider().force_flush()
        spans = TraceStore(str(fresh_tracing)).get_trace(result.trace_id).spans
        by_name = {s.name: s for s in spans}
        root = by_name["supervisor.desk"]
        assert root.attributes["supervisor.routing"] == "decisions"
        assert root.attributes["supervisor.route.worker"] == "product_enquiry"
        assert root.attributes["supervisor.route.fallback"] is False
        assert root.attributes["fastaiagent.runner.type"] == "supervisor"
        decide = next(s for s in spans if s.name == "llm.custom.decisions.gpt-6-luna")
        worker = next(s for s in spans if s.name == "agent.w-product_enquiry")
        assert decide.parent_span_id == root.span_id
        assert worker.parent_span_id == root.span_id


class TestResume:
    def test_a_paused_worker_resumes_without_re_routing(self, decisions_stub, tmp_path) -> None:
        cp = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))

        def approve_refund(order_id: str) -> str:
            """Refund an order (needs a manager's approval)."""
            decision = interrupt("refund needs approval", {"order_id": order_id})
            return "refunded" if decision.approved else "refund declined"

        def worker_llm(messages):
            tool_msgs = [m for m in messages if m.role == MessageRole.tool]
            if not tool_msgs:
                return "", [{"name": "approve_refund_rt", "arguments": {"order_id": "A1"}}]
            return f"Done: {tool_msgs[-1].content}"

        complaints = Worker(
            agent=Agent(
                name="w-complaint",
                llm=FunctionModel(worker_llm),
                tools=[FunctionTool(name="approve_refund_rt", fn=approve_refund)],
            ),
            role="complaint",
            description="Complaints and refunds.",
        )
        team = Supervisor(
            name="desk-rt",
            llm=TestModel(),
            workers=[complaints, _worker("other", "general help")],
            routing="decisions",
            router_llm=decisions_stub.client(),
            fallback_worker="other",
            checkpointer=cp,
        )
        decisions_stub.choices["worker"] = "complaint"
        exec_id = str(uuid.uuid4())
        first = team.run("refund A1 please", execution_id=exec_id)
        assert first.status == "paused"
        routed_calls = len(_decisions(decisions_stub))

        # The router is down now: a resume must not ask it again.
        decisions_stub.stop()
        resumed = team.resume(exec_id, resume_value=Resume(approved=True))
        assert resumed.status == "completed"
        assert resumed.output == "Done: refunded"
        assert resumed.route.worker == "complaint"
        assert len(_decisions(decisions_stub)) == routed_calls


class TestConstruction:
    @pytest.mark.parametrize(
        "kw, match",
        [
            ({"routing": "decision"}, "routing must be"),
            ({"routing": "decisions", "fallback_worker": None}, "fallback_worker"),
            ({"routing": "decisions", "fallback_worker": "nobody"}, "fallback_worker"),
            ({"routing": "decisions", "fallback_worker": "a", "routing_min_confidence": 2}, "0..1"),
            (
                {"routing": "decisions", "fallback_worker": "a", "routing_questions": {"worker": Predicate(instructions="x")}},
                "worker",
            ),
        ],
    )
    def test_a_router_that_could_not_route_fails_at_construction(self, kw, match) -> None:
        with pytest.raises(ValueError, match=match):
            Supervisor(
                name="s",
                llm=TestModel(),
                workers=[_worker("a", "x"), _worker("b", "y")],
                **kw,
            )

    def test_needs_two_distinct_workers(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            Supervisor(name="s", llm=TestModel(), workers=[_worker("a", "x")], routing="decisions", fallback_worker="a")
        with pytest.raises(ValueError, match="unique"):
            Supervisor(
                name="s",
                llm=TestModel(),
                workers=[_worker("a", "x"), _worker("a", "y")],
                routing="decisions",
                fallback_worker="a",
            )

    def test_tools_routing_is_the_default_and_unchanged(self) -> None:
        sup = Supervisor(name="s", llm=TestModel(), workers=[_worker("a", "x")])
        assert sup.routing == "tools"
        assert "routing" not in sup.to_dict()

    def test_to_dict(self, decisions_stub) -> None:
        d = _team(decisions_stub).to_dict()
        assert d["routing"] == "decisions" and d["fallback_worker"] == "other"
