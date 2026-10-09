"""``LLMClient.decide`` / ``adecide`` — OpenAI's Decisions API (1.84.0).

No mocks: the real client (httpx transport, retries, pricing) runs against the
in-process ``/v1/decisions`` stand-in in ``tests/_decisions_stub.py``, and the
injected-client cases use the real ``openai`` package. Shapes follow OpenAI's own
published types (``openai`` 3.26.0) and a live response captured 2026-10-09.
"""

from __future__ import annotations

import asyncio

import pytest

from fastaiagent._internal.errors import LLMError, LLMProviderError
from fastaiagent._internal.pricing import run_cost, start_run_cost
from fastaiagent.llm import (
    Choice,
    ChoiceAnswer,
    DecisionResult,
    Level,
    LLMClient,
    Option,
    Predicate,
    PredicateAnswer,
    Refusal,
    Score,
    ScoreAnswer,
    SystemMessage,
    UserMessage,
)
from fastaiagent.llm.decisions import (
    build_decision_input,
    normalize_questions,
    parse_decision,
    question_from_dict,
)
from fastaiagent.multimodal.image import Image
from fastaiagent.testing import FunctionModel, TestModel
from tests._decisions_stub import INPUT_TOKENS

# A 1x1 PNG — real image bytes, so ``maybe_resize`` and base64 run for real.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360f8cfc0000003010100c9fe92ef0000000049454e44ae426082"
)

DEPT = Choice(
    name="department",
    instructions="Which department should handle this?",
    options={"billing": "Payments and refunds", "technical": None, "other": None},
)
DUP = Predicate(name="duplicate", instructions="The customer reports a duplicate charge.")
SEV = Score(name="severity", instructions="How severe?", levels=["Cosmetic", "Workaround", "Blocked"])


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


class TestQuestions:
    def test_wire_shapes_follow_openais_types(self) -> None:
        assert DUP.to_wire() == {
            "type": "predicate",
            "instructions": "The customer reports a duplicate charge.",
            "name": "duplicate",
        }
        assert DEPT.to_wire()["choices"] == [
            {"value": "billing", "description": "Payments and refunds"},
            {"value": "technical"},
            {"value": "other"},
        ]
        assert SEV.to_wire()["levels"] == [
            {"label": "Cosmetic"},
            {"label": "Workaround"},
            {"label": "Blocked"},
        ]

    def test_an_unnamed_question_sends_no_name(self) -> None:
        assert "name" not in Predicate(instructions="x").to_wire()

    def test_choice_values_stay_typed(self) -> None:
        q = Choice(instructions="Complaint?", options=[True, False, "true"])
        values = [c["value"] for c in q.to_wire()["choices"]]
        assert values == [True, False, "true"]
        assert [type(v) for v in values] == [bool, bool, str]

    def test_options_accept_option_objects_and_the_wire_alias(self) -> None:
        a = Choice(instructions="x", options=[Option(value="a"), Option(value="b", description="B")])
        b = question_from_dict(a.to_wire())
        assert isinstance(b, Choice) and b.options == a.options
        c = Score(instructions="x", levels=[Level(label="lo"), Level(label="hi", description="H")])
        assert question_from_dict(c.to_wire()) == c

    def test_dump_round_trips(self) -> None:
        for q in (DUP, DEPT, SEV):
            assert question_from_dict(q.model_dump()) == q

    @pytest.mark.parametrize(
        "build, msg",
        [
            (lambda: Predicate(instructions="  "), "non-empty"),
            (lambda: Choice(instructions="x", options=["only"]), "at least 2 options"),
            (lambda: Choice(instructions="x", options=["a", "a"]), "listed twice"),
            (lambda: Choice(instructions="x", options="ab"), "list of values"),
            (lambda: Score(instructions="x", levels=["one"]), "at least 2 levels"),
            (lambda: Score(instructions="x", levels=["a", "a"]), "unique"),
            (lambda: Score(instructions="x", levels=["a", " "]), "non-empty"),
        ],
    )
    def test_a_question_that_cannot_decide_anything_is_refused(self, build, msg) -> None:
        # CLAUDE.md §2.4: an unusable question raises — it never reaches the
        # endpoint, so it can never come back as a clean answer.
        with pytest.raises(ValueError, match=msg):
            build()

    def test_a_string_and_a_bool_are_distinct_options(self) -> None:
        q = Choice(instructions="x", options=["true", True])
        assert len(q.options) == 2

    def test_a_name_to_question_mapping(self) -> None:
        qs = normalize_questions(
            {
                "urgent": Predicate(instructions="Is this urgent?"),
                "dept": {"type": "choice", "instructions": "x", "choices": [{"value": "a"}, {"value": "b"}]},
            }
        )
        assert [q.name for q in qs] == ["urgent", "dept"]
        assert isinstance(qs[1], Choice)
        with pytest.raises(ValueError, match="already named"):
            normalize_questions({"a": Predicate(name="b", instructions="x")})
        with pytest.raises(TypeError):
            normalize_questions({"a": "not a question"})

    def test_normalize(self) -> None:
        assert normalize_questions(DUP) == [DUP]
        assert normalize_questions([DUP, DEPT.to_wire()]) == [DUP, DEPT]
        with pytest.raises(ValueError, match="at least one question"):
            normalize_questions([])
        with pytest.raises(ValueError, match="unique"):
            normalize_questions([DUP, Predicate(name="duplicate", instructions="y")])
        with pytest.raises(TypeError):
            normalize_questions(["not a question"])


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class TestInput:
    def test_text(self) -> None:
        assert build_decision_input("hi", image_cap_mb=18) == "hi"

    def test_image_becomes_an_inline_data_url(self) -> None:
        out = build_decision_input(
            ["look", Image.from_bytes(PNG, "image/png")], image_cap_mb=18
        )
        assert out[0]["role"] == "user"
        text, image = out[0]["content"]
        assert text == {"type": "input_text", "text": "look"}
        assert image["type"] == "input_image"
        assert image["image_url"].startswith("data:image/png;base64,")
        assert "detail" not in image  # "auto" is the endpoint's default

    def test_image_detail_passes_through(self) -> None:
        out = build_decision_input(Image.from_bytes(PNG, "image/png", detail="low"), image_cap_mb=18)
        assert out[0]["content"][0]["detail"] == "low"

    def test_user_messages(self) -> None:
        out = build_decision_input(
            [UserMessage("first"), UserMessage(["second", Image.from_bytes(PNG, "image/png")])],
            image_cap_mb=18,
        )
        assert out[0] == {"role": "user", "content": "first"}
        assert out[1]["content"][1]["type"] == "input_image"

    def test_non_user_messages_are_rejected(self) -> None:
        with pytest.raises(LLMError, match="system message"):
            build_decision_input([SystemMessage("be nice")], image_cap_mb=18)

    def test_pdf_is_rejected(self) -> None:
        from fastaiagent.multimodal.pdf import PDF

        with pytest.raises(LLMError, match="PDF"):
            build_decision_input(["read this", PDF(data=b"%PDF-1.4", text="x")], image_cap_mb=18)


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------

LIVE_SHAPE = {
    # Captured from the live endpoint on 2026-10-09 (trimmed).
    "model": "gpt-6-luna",
    "answers": [
        {
            "type": "choice",
            "name": "department",
            "choice": "billing",
            "probabilities": [
                {"value": "billing", "probability": 1.0},
                {"value": "technical", "probability": 0.0},
                {"value": "other", "probability": 0.0},
            ],
            "confidence": 1.0,
        },
        {"type": "predicate", "name": "duplicate", "probability": 1.0},
        {
            "type": "score",
            "name": "severity",
            "score": 1.25,
            "probabilities": [
                {"value": 0, "label": "Cosmetic", "probability": 0.05},
                {"value": 1, "label": "Workaround", "probability": 0.65},
                {"value": 2, "label": "Blocked", "probability": 0.3},
            ],
            "confidence": 0.48,
        },
    ],
    "usage": {"input_tokens": 641, "output_tokens": 0, "total_tokens": 641},
}


class TestAnswers:
    def test_parses_the_live_shape(self) -> None:
        r = parse_decision(LIVE_SHAPE, [DEPT, DUP, SEV])
        assert isinstance(r["department"], ChoiceAnswer) and r["department"].choice == "billing"
        assert isinstance(r[1], PredicateAnswer) and r[1].probability == 1.0
        sev = r["severity"]
        assert isinstance(sev, ScoreAnswer)
        assert sev.score == 1.25
        assert sev.normalized == pytest.approx(0.625)  # 1.25 / (3 - 1)
        assert sev.level == "Workaround"
        assert r.model == "gpt-6-luna" and r.usage["input_tokens"] == 641
        assert not r.refused and len(r) == 3
        assert r.predicates == {"duplicate": r[1]}
        assert r.choices == {"department": r[0]}
        assert r.scores == {"severity": r[2]}

    def test_typed_choice_lookup(self) -> None:
        q = Choice(name="c", instructions="x", options=[True, False])
        r = parse_decision(
            {
                "answers": [
                    {
                        "type": "choice",
                        "name": "c",
                        "choice": True,
                        "probabilities": [
                            {"value": True, "probability": 0.99},
                            {"value": False, "probability": 0.01},
                        ],
                        "confidence": 0.98,
                    }
                ]
            },
            [q],
        )
        ans = r["c"]
        assert ans.choice is True
        assert ans.probability_of(True) == 0.99
        with pytest.raises(KeyError):
            ans.probability_of("true")

    def test_a_refusal_is_its_own_answer(self) -> None:
        r = parse_decision(
            {"answers": [{"type": "refusal", "name": "duplicate"}, LIVE_SHAPE["answers"][2]]},
            [DUP, SEV],
        )
        assert isinstance(r["duplicate"], Refusal)
        assert r.refused and r.refusals == [r[0]]
        assert isinstance(r["severity"], ScoreAnswer)

    def test_unnamed_answers_index_by_position(self) -> None:
        r = parse_decision(
            {"answers": [{"type": "predicate", "name": None, "probability": 0.2}]},
            [Predicate(instructions="x")],
        )
        assert r[0].probability == 0.2
        assert r.get("missing") is None

    @pytest.mark.parametrize(
        "data, questions, match",
        [
            ({}, [DUP], "no 'answers'"),
            ({"answers": []}, [DUP], "answered 0 question"),
            ({"answers": [{"type": "verdict", "name": "duplicate"}]}, [DUP], "verdict"),
            (
                {"answers": [{"type": "predicate", "name": "department", "probability": 1}]},
                [DEPT],
                "with a predicate answer",
            ),
        ],
    )
    def test_an_unreadable_reply_raises_never_guesses(self, data, questions, match) -> None:
        with pytest.raises(LLMError, match=match):
            parse_decision(data, questions)


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


class TestDecide:
    def test_round_trip(self, decisions_stub) -> None:
        decisions_stub.probabilities["duplicate"] = 0.83
        r = decisions_stub.client().decide("I was charged twice.", [DEPT, DUP, SEV])
        sent = decisions_stub.requests[-1]
        assert sent["path"] == "/v1/decisions"
        assert sent["auth"] == "Bearer stub-key"
        assert sent["body"]["model"] == "gpt-6-luna"
        assert sent["body"]["input"] == "I was charged twice."
        assert [q["type"] for q in sent["body"]["questions"]] == ["choice", "predicate", "score"]
        assert "safety_identifier" not in sent["body"]
        assert r["department"].choice == "billing"
        assert r["duplicate"].probability == 0.83
        assert r["severity"].level == "Blocked"
        assert r.latency_ms >= 0
        assert r.request_id == "req_stub_1"

    def test_questions_as_a_mapping(self, decisions_stub) -> None:
        r = decisions_stub.client().decide(
            "x", {"urgent": Predicate(instructions="Is this urgent?"), "dept": DEPT.model_copy(update={"name": None})}
        )
        names = [q["name"] for q in decisions_stub.requests[-1]["body"]["questions"]]
        assert names == ["urgent", "dept"]
        assert r.predicates["urgent"].probability == 0.9
        assert r.choices["dept"].choice == "billing"

    def test_async_and_safety_identifier(self, decisions_stub) -> None:
        r = asyncio.run(
            decisions_stub.client().adecide("x", DUP, safety_identifier="user-123")
        )
        assert decisions_stub.requests[-1]["body"]["safety_identifier"] == "user-123"
        assert isinstance(r, DecisionResult)

    def test_cost_is_input_only_at_the_decisions_rate(self, decisions_stub) -> None:
        r = decisions_stub.client().decide("x", DUP)
        # $0.10 per 1M input tokens, nothing for output.
        assert r.cost_usd == pytest.approx(INPUT_TOKENS * 0.10 / 1_000_000)

    def test_decisions_count_toward_the_run_cost(self, decisions_stub) -> None:
        async def run() -> tuple[float, bool]:
            start_run_cost()
            await decisions_stub.client().adecide("x", DUP)
            await decisions_stub.client().adecide("y", DUP)
            return run_cost()

        usd, known = asyncio.run(run())
        assert known and usd == pytest.approx(2 * INPUT_TOKENS * 0.10 / 1_000_000)

    def test_an_unpriced_model_has_no_cost(self, decisions_stub) -> None:
        r = decisions_stub.client(model="gpt-7-someday").decide("x", DUP)
        assert r.cost_usd is None

    def test_5xx_is_retried(self, decisions_stub) -> None:
        decisions_stub.script = [(503, {"error": {"message": "busy"}})]
        r = decisions_stub.client(max_retries=1).decide("x", DUP)
        assert len(decisions_stub.requests) == 2
        assert r["duplicate"].probability == 0.9

    def test_4xx_is_not_retried_and_keeps_openais_message(self, decisions_stub) -> None:
        decisions_stub.script = [
            (
                400,
                {
                    "error": {
                        "message": "Invalid 'questions': empty array.",
                        "type": "invalid_request_error",
                        "code": "empty_array",
                    }
                },
            )
        ]
        with pytest.raises(LLMProviderError, match="empty_array") as err:
            decisions_stub.client(max_retries=3).decide("x", DUP)
        assert err.value.status_code == 400
        assert len(decisions_stub.requests) == 1

    def test_non_json_reply(self, decisions_stub) -> None:
        decisions_stub.script = [(200, b"<html>gateway</html>")]
        with pytest.raises(LLMProviderError, match="non-JSON"):
            decisions_stub.client().decide("x", DUP)

    @pytest.mark.parametrize("provider", ["anthropic", "ollama", "bedrock", "groq"])
    def test_providers_that_do_not_serve_the_endpoint(self, provider) -> None:
        with pytest.raises(LLMError, match="Decisions API"):
            LLMClient(provider=provider, model="x", api_key="k").decide("x", DUP)

    def test_custom_needs_a_base_url(self) -> None:
        with pytest.raises(LLMError, match="base_url"):
            LLMClient(provider="custom", model="gpt-6-luna", api_key="k").decide("x", DUP)

    def test_missing_api_key(self, decisions_stub, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = decisions_stub.client(api_key=None)
        with pytest.raises(LLMProviderError, match="OPENAI_API_KEY"):
            client.decide("x", DUP)
        assert decisions_stub.requests == []

    def test_a_bad_question_never_reaches_the_endpoint(self, decisions_stub) -> None:
        with pytest.raises(ValueError):
            decisions_stub.client().decide("x", [{"type": "choice", "instructions": "x", "choices": []}])
        assert decisions_stub.requests == []


class TestInjectedOpenAIClient:
    """``openai_client=`` — the real ``openai`` package, whatever version is installed.

    openai < 3.26 has no ``client.decisions``; the generic ``client.post`` path
    runs, reusing the client's base_url and auth. (The ``.decisions.create``
    path is exercised against openai 3.26 by hand — see the 1.84.0 notes — since
    the dev install pins openai < 3.)
    """

    def test_sync_client(self, decisions_stub) -> None:
        openai = pytest.importorskip("openai")
        oc = openai.OpenAI(api_key="injected", base_url=decisions_stub.base_url, max_retries=0)
        llm = LLMClient(model="gpt-6-luna", openai_client=oc)
        r = llm.decide("x", [DUP, DEPT])
        assert decisions_stub.requests[-1]["auth"] == "Bearer injected"
        assert decisions_stub.requests[-1]["path"] == "/v1/decisions"
        assert r["department"].choice == "billing"
        assert r.request_id == f"req_stub_{len(decisions_stub.requests)}"

    def test_async_client(self, decisions_stub) -> None:
        openai = pytest.importorskip("openai")
        oc = openai.AsyncOpenAI(api_key="injected", base_url=decisions_stub.base_url, max_retries=0)
        r = asyncio.run(LLMClient(model="gpt-6-luna", openai_client=oc).adecide("x", SEV))
        assert r["severity"].level == "Blocked"

    def test_errors_are_wrapped_and_retried(self, decisions_stub) -> None:
        openai = pytest.importorskip("openai")
        oc = openai.OpenAI(api_key="injected", base_url=decisions_stub.base_url, max_retries=0)
        decisions_stub.script = [(500, {"error": {"message": "boom"}})]
        r = LLMClient(model="gpt-6-luna", openai_client=oc, max_retries=1).decide("x", DUP)
        assert r["duplicate"].probability == 0.9
        decisions_stub.script = [(401, {"error": {"message": "bad key"}})]
        with pytest.raises(LLMProviderError) as err:
            LLMClient(model="gpt-6-luna", openai_client=oc).decide("x", DUP)
        assert err.value.status_code == 401


# ---------------------------------------------------------------------------
# Offline doubles
# ---------------------------------------------------------------------------


class TestOfflineModels:
    def test_test_model_serves_canned_decisions_in_order(self) -> None:
        m = TestModel(
            decisions=[
                {"answers": [{"type": "predicate", "name": "duplicate", "probability": 0.1}]},
                {"answers": [{"type": "predicate", "name": "duplicate", "probability": 0.7}]},
            ]
        )
        assert m.decide("a", DUP)["duplicate"].probability == 0.1
        assert m.decide("b", DUP)["duplicate"].probability == 0.7
        assert m.decide("c", DUP)["duplicate"].probability == 0.7  # last repeats
        assert [c["input"] for c in m.decision_calls] == ["a", "b", "c"]

    def test_test_model_without_canned_decisions_refuses_to_invent_one(self) -> None:
        with pytest.raises(ValueError, match="no canned decision"):
            TestModel().decide("x", DUP)

    def test_a_canned_decision_must_match_the_questions(self) -> None:
        m = TestModel(decisions={"answers": [{"type": "predicate", "probability": 0.5}]})
        with pytest.raises(LLMError, match="answered 1 question"):
            m.decide("x", [DUP, SEV])

    def test_function_model(self) -> None:
        def decide_fn(input, questions):
            p = 0.95 if "twice" in input else 0.05
            return DecisionResult(answers=[PredicateAnswer(name=questions[0].name, probability=p)])

        m = FunctionModel(lambda msgs: "ok", decide_fn=decide_fn)
        assert m.decide("charged twice", DUP)["duplicate"].probability == 0.95
        assert m.decide("all good", DUP)["duplicate"].probability == 0.05

    def test_function_model_without_decide_fn(self) -> None:
        with pytest.raises(ValueError, match="decide_fn"):
            FunctionModel(lambda msgs: "ok").decide("x", DUP)
