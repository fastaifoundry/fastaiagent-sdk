"""Decisions API routing (1.84.0): Chain decision nodes, ``decision_tool``, and
Supervisor ``validation_mode="decisions"``.

No mocks: decisions go to the local ``/v1/decisions`` stand-in through the real
``LLMClient``; chat turns come from ``TestModel`` / ``FunctionModel``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from fastaiagent.agent import Agent
from fastaiagent.agent.team import Supervisor, Worker
from fastaiagent.chain import Chain
from fastaiagent.chain.node import NodeType
from fastaiagent.llm import Choice, Predicate, Score
from fastaiagent.llm.message import MessageRole
from fastaiagent.testing import FunctionModel, TestModel
from fastaiagent.tool import decision_tool

TRIAGE = Choice(
    instructions="Which team should handle this ticket?",
    options={"billing": "Payments and refunds", "technical": "Bugs and outages"},
)


def _agent(name: str) -> Agent:
    return Agent(name=name, llm=TestModel(response=f"{name} handled it"))


def _triage_chain(stub, **decision) -> Chain:
    chain = Chain("triage", checkpoint_enabled=False)
    chain.add_node(
        "triage",
        type=NodeType.condition,
        decision={
            "question": TRIAGE,
            "input": "{{input.message}}",
            "llm": stub.llm_kwargs(),
            **decision,
        },
    )
    for name in ("billing_agent", "tech_agent", "human"):
        chain.add_node(name, agent=_agent(name))
    chain.connect("triage", "billing_agent", label="billing")
    chain.connect("triage", "tech_agent", label="technical")
    chain.connect("triage", "human")  # default
    return chain


class TestChainDecisionNode:
    def test_routes_on_the_chosen_option(self, decisions_stub) -> None:
        decisions_stub.choices[None] = "technical"
        result = asyncio.run(
            _triage_chain(decisions_stub).aexecute({"message": "The app crashes on login."})
        )
        assert "tech_agent" in result.node_results
        assert "billing_agent" not in result.node_results
        assert "human" not in result.node_results
        assert result.node_results["triage"]["matched"] == "technical"
        assert result.node_results["triage"]["decision"]["confidence"] == 1.0
        assert decisions_stub.requests[-1]["body"]["input"] == "The app crashes on login."

    def test_a_refusal_routes_default(self, decisions_stub) -> None:
        decisions_stub.refuse.add(None)
        result = asyncio.run(_triage_chain(decisions_stub).aexecute({"message": "x"}))
        assert "human" in result.node_results
        assert result.node_results["triage"]["decision"]["refused"] is True

    def test_low_confidence_routes_default(self, decisions_stub) -> None:
        decisions_stub.script = [
            (
                200,
                {
                    "model": "gpt-6-luna",
                    "answers": [
                        {
                            "type": "choice",
                            "name": None,
                            "choice": "billing",
                            "probabilities": [
                                {"value": "billing", "probability": 0.55},
                                {"value": "technical", "probability": 0.45},
                            ],
                            "confidence": 0.1,
                        }
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            )
        ]
        result = asyncio.run(
            _triage_chain(decisions_stub, min_confidence=0.6).aexecute({"message": "hmm"})
        )
        assert "human" in result.node_results
        assert "billing_agent" not in result.node_results

    def test_boolean_options_route_on_true_false_labels(self, decisions_stub) -> None:
        chain = Chain("yesno", checkpoint_enabled=False)
        chain.add_node(
            "gate",
            type=NodeType.condition,
            decision={
                "question": Choice(instructions="Is this a refund request?", options=[True, False]),
                "llm": decisions_stub.llm_kwargs(),
            },
        )
        chain.add_node("refunds", agent=_agent("refunds"))
        chain.add_node("other", agent=_agent("other"))
        chain.connect("gate", "refunds", label="True")
        chain.connect("gate", "other", label="False")
        chain.connect("gate", "other")
        decisions_stub.choices[None] = True
        result = asyncio.run(chain.aexecute({"input": "refund me"}))
        assert "refunds" in result.node_results

    def test_the_chain_round_trips_through_to_dict(self, decisions_stub) -> None:
        chain = _triage_chain(decisions_stub)
        data = json.loads(json.dumps(chain.to_dict()))  # plain JSON all the way down
        decision = data["nodes"][0]["config"]["decision"]
        assert decision["question"]["type"] == "choice"
        assert decision["min_confidence"] == 0.0
        rebuilt = Chain.from_dict(data)
        assert rebuilt.nodes[0].config["decision"] == chain.nodes[0].config["decision"]

    def test_every_option_needs_an_edge(self, decisions_stub) -> None:
        chain = Chain("missing-edge", checkpoint_enabled=False)
        chain.add_node(
            "triage",
            type=NodeType.condition,
            decision={"question": TRIAGE, "llm": decisions_stub.llm_kwargs()},
        )
        chain.add_node("billing_agent", agent=_agent("billing_agent"))
        chain.add_node("human", agent=_agent("human"))
        chain.connect("triage", "billing_agent", label="billing")
        chain.connect("triage", "human")
        errors = chain.validate()
        assert any("'technical'" in e for e in errors), errors

    @pytest.mark.parametrize(
        "decision, exc, match",
        [
            ({"question": Predicate(instructions="x")}, ValueError, "routes on a Choice"),
            ({"question": TRIAGE, "llm": object()}, TypeError, "kwargs"),
            ({"question": TRIAGE, "min_confidence": 2}, ValueError, "0..1"),
            ({"question": TRIAGE, "bogus": 1}, ValueError, "unknown decision key"),
            ({}, ValueError, "needs a Choice"),
        ],
    )
    def test_a_decision_that_cannot_route_fails_at_build(self, decision, exc, match) -> None:
        chain = Chain("bad", checkpoint_enabled=False)
        with pytest.raises(exc, match=match):
            chain.add_node("triage", type=NodeType.condition, decision=decision)

    def test_choice_shorthand(self, decisions_stub) -> None:
        chain = Chain("short", checkpoint_enabled=False)
        chain.add_node("triage", type=NodeType.condition, decision=TRIAGE)
        assert chain.nodes[0].config["decision"]["input"] == "{{input}}"


class TestDecisionTool:
    def test_an_agent_calls_it(self, decisions_stub) -> None:
        decisions_stub.choices["department"] = "billing"
        decisions_stub.probabilities["urgent"] = 0.81
        tool = decision_tool(
            {
                "department": TRIAGE,
                "urgent": Predicate(instructions="The customer needs help right now."),
            },
            llm=decisions_stub.client(),
            name="triage_ticket",
        )
        assert tool.parameters["required"] == ["input"]
        assert "department: Which team" in tool.description

        seen: list[str] = []

        def chat(messages):
            tool_msgs = [m for m in messages if m.role == MessageRole.tool]
            if not tool_msgs:
                return "", [{"name": "triage_ticket", "arguments": {"input": "charged twice!"}}]
            seen.append(str(tool_msgs[-1].content))
            return "Routed to billing."

        result = Agent(name="support", llm=FunctionModel(chat), tools=[tool]).run("help")
        assert result.output == "Routed to billing."
        answers = json.loads(seen[0])
        assert answers["department"]["choice"] == "billing"
        assert answers["urgent"] == {"probability": 0.81}
        assert decisions_stub.requests[-1]["body"]["input"] == "charged twice!"

    def test_score_and_refusal_shapes(self, decisions_stub) -> None:
        decisions_stub.refuse.add("mood")
        tool = decision_tool(
            {
                "sev": Score(instructions="Severity", levels=["low", "high"]),
                "mood": Predicate(instructions="Angry"),
            },
            llm=decisions_stub.client(),
        )
        out = asyncio.run(tool.aexecute({"input": "x"})).output
        assert out["sev"]["level"] == "high" and out["sev"]["normalized"] == 1.0
        assert out["mood"] == {"refused": True}

    def test_a_single_unnamed_question_is_named_answer(self, decisions_stub) -> None:
        tool = decision_tool(Predicate(instructions="x"), llm=decisions_stub.client())
        assert asyncio.run(tool.aexecute({"input": "y"})).output == {"answer": {"probability": 0.9}}

    def test_unnamed_questions_in_a_list_are_refused(self) -> None:
        with pytest.raises(ValueError, match="named"):
            decision_tool([Predicate(instructions="a"), Predicate(instructions="b")])

    def test_works_offline_with_test_model(self) -> None:
        tool = decision_tool(
            Predicate(name="spam", instructions="This is spam."),
            llm=TestModel(decisions={"answers": [{"type": "predicate", "name": "spam", "probability": 0.99}]}),
        )
        assert asyncio.run(tool.aexecute({"input": "WIN $$$"})).output == {"spam": {"probability": 0.99}}


def _predicate_reply(p: float) -> tuple[int, dict]:
    return 200, {
        "model": "gpt-6-luna",
        "answers": [{"type": "predicate", "name": "approved", "probability": p}],
        "usage": {"input_tokens": 50, "output_tokens": 0},
    }


class TestSupervisorDecisionValidation:
    def _supervisor(self, stub, worker_outputs, **kw) -> tuple[Supervisor, list]:
        worker_inputs: list = []
        state = {"n": 0}

        def worker_fn(messages):
            worker_inputs.append(messages)
            out = worker_outputs[min(state["n"], len(worker_outputs) - 1)]
            state["n"] += 1
            return out

        worker = Worker(
            agent=Agent(name="w-alpha", llm=FunctionModel(worker_fn)),
            role="alpha",
            description="does things",
        )
        sup_state = {"n": 0}

        def sup_fn(messages):
            sup_state["n"] += 1
            if sup_state["n"] == 1:
                return "", [{"name": "delegate_to_alpha", "arguments": {"task": "summarise Q3"}}]
            return "done"

        sup = Supervisor(
            name="s",
            llm=FunctionModel(sup_fn),
            workers=[worker],
            validate_outputs=True,
            validation_mode="decisions",
            validation_llm=stub.client(),
            **kw,
        )
        return sup, worker_inputs

    def test_reject_then_accept(self, decisions_stub) -> None:
        decisions_stub.script = [_predicate_reply(0.2), _predicate_reply(0.9)]
        sup, worker_inputs = self._supervisor(decisions_stub, ["meh", "Q3 revenue grew 12%."])
        assert sup.run("go").output == "done"
        decision_calls = [r for r in decisions_stub.requests if r["path"].endswith("/decisions")]
        assert len(decision_calls) == 2
        # The task and output travel as evidence; the criterion is fixed.
        assert "Original task:\nsummarise Q3" in decision_calls[0]["body"]["input"]
        assert "Worker output:\nmeh" in decision_calls[0]["body"]["input"]
        # The worker's retry carried the reviewer's feedback.
        retry_text = json.dumps([m.content for m in worker_inputs[1]], default=str)
        assert "p=0.20" in retry_text

    def test_threshold(self, decisions_stub) -> None:
        decisions_stub.script = [_predicate_reply(0.7)]
        sup, _ = self._supervisor(decisions_stub, ["fine"], validation_threshold=0.6)
        sup.run("go")
        assert len([r for r in decisions_stub.requests if r["path"].endswith("/decisions")]) == 1

    def test_refusal_and_failure_fail_open(self, decisions_stub) -> None:
        decisions_stub.refuse.add("approved")
        sup, worker_inputs = self._supervisor(decisions_stub, ["fine"])
        assert sup.run("go").output == "done"
        assert len(worker_inputs) == 1  # approved, not retried

        decisions_stub.refuse.clear()
        decisions_stub.script = [(500, {"error": {"message": "down"}})]
        sup, worker_inputs = self._supervisor(decisions_stub, ["fine"])
        assert sup.run("go").output == "done"
        assert len(worker_inputs) == 1

    def test_bad_settings(self) -> None:
        with pytest.raises(ValueError, match="validation_mode"):
            Supervisor(name="s", llm=TestModel(), validation_mode="decision")
        with pytest.raises(ValueError, match="validation_threshold"):
            Supervisor(name="s", llm=TestModel(), validation_threshold=1.2)

    def test_to_dict(self, decisions_stub) -> None:
        sup, _ = self._supervisor(decisions_stub, ["x"])
        assert sup.to_dict()["validation_mode"] == "decisions"
        assert "validation_mode" not in Supervisor(name="s", llm=TestModel()).to_dict()
