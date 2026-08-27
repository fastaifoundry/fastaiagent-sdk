"""LLM judge — template alias rendering, tolerant verdict parsing, scale/threshold.

Deterministic and mock-free: the LLM-backed cases drive ``TestModel``/``FunctionModel``,
the real ``LLMClient`` subclasses the SDK ships, so a canned judge reply travels the
same code path a provider reply would. Real-LLM coverage lives in
``tests/e2e/test_eval_judge_aliases_e2e.py``.
"""

import json
import warnings

import pytest

from fastaiagent.eval import GEval, LLMJudge
from fastaiagent.eval.llm_judge import (
    _RETRY_INSTRUCTION,
    _parse_verdict,
    render_judge_template,
)
from fastaiagent.testing import FunctionModel, TestModel


def _judge(response, **kw):
    """An LLMJudge whose judge model returns ``response`` (str, or list to round-robin)."""
    return LLMJudge(llm=TestModel(response=response), **kw)


class TestRenderJudgeTemplate:
    """The template is a wire contract with the control plane: every alias interpolates."""

    def test_canonical_placeholders(self):
        out = render_judge_template("Q: {input} A: {output} E: {expected}", "q", "a", "e")
        assert out == "Q: q A: a E: e"

    def test_expected_output_alias(self):
        assert render_judge_template("E: {expected_output}", "q", "a", "e") == "E: e"

    def test_double_brace_aliases(self):
        out = render_judge_template(
            "Q: {{input}} A: {{output}} E1: {{expected}} E2: {{expected_output}}", "q", "a", "e"
        )
        assert out == "Q: q A: a E1: e E2: e"

    def test_mixed_aliases(self):
        out = render_judge_template("{input} {expected_output} {{output}}", "q", "a", "e")
        assert out == "q e a"

    def test_double_and_single_brace_both_present(self):
        """``{{input}}`` contains ``{input}`` — order must not leave stray braces."""
        out = render_judge_template("{{input}} then {input}", "q", "a", "e")
        assert out == "q then q"
        assert "{" not in out and "}" not in out

    def test_literal_json_exemplar_untouched(self):
        """Templates carry JSON examples — this is why we never use str.format()."""
        tpl = 'Score {input}. Respond with {"score": <number>, "reasoning": "<why>"}'
        out = render_judge_template(tpl, "q", "a", "e")
        assert '{"score": <number>, "reasoning": "<why>"}' in out

    def test_no_placeholder_template_unchanged(self):
        assert render_judge_template("no placeholders", "q", "a", "e") == "no placeholders"

    def test_single_pass_does_not_rescan_substituted_content(self):
        """Content that *looks* like a placeholder must survive as literal text."""
        out = render_judge_template("{input} | {output}", "say {output}", "ACTUAL", "e")
        assert out == "say {output} | ACTUAL"

    def test_unknown_placeholder_left_alone(self):
        assert render_judge_template("{context}", "q", "a", "e") == "{context}"


class TestParseVerdict:
    """Pure parsing — JSON, fenced JSON, prose-with-embedded-score, unparseable."""

    def test_clean_json(self):
        assert _parse_verdict('{"score": 0.9, "reasoning": "good"}') == (0.9, "good")

    def test_fenced_json(self):
        raw = '```json\n{"score": 1, "reasoning": "ok"}\n```'
        assert _parse_verdict(raw) == (1.0, "ok")

    def test_prose_with_embedded_score(self):
        raw = 'Here is my verdict: "score": 0.6, "reasoning": "partly right". Thanks!'
        assert _parse_verdict(raw) == (0.6, "partly right")

    def test_prose_score_without_reasoning(self):
        score, reason = _parse_verdict('the answer is "score": 0.25')
        assert score == 0.25
        assert reason == "Could not parse reasoning"

    def test_json_without_score_key_is_unparseable(self):
        """Previously became a silent 0.0 — now it earns a retry."""
        assert _parse_verdict('{"reasoning": "I forgot the score"}') is None

    def test_garbage_is_unparseable(self):
        assert _parse_verdict("I am not going to answer that.") is None

    def test_empty_is_unparseable(self):
        assert _parse_verdict("") is None


class TestJudgeParsePaths:
    """End-to-end through LLMJudge with a canned judge model."""

    def test_clean_json_single_call(self):
        judge = _judge(json.dumps({"score": 1.0, "reasoning": "correct"}))
        result = judge.score(input="2+2?", output="4", expected="4")
        assert result.score == 1.0
        assert result.passed
        assert result.reason == "correct"
        assert len(judge._llm.calls) == 1

    def test_prose_reply_uses_regex_fallback_not_zero(self):
        judge = _judge('I think this is right. "score": 0.6, "reasoning": "close enough"')
        result = judge.score(input="2+2?", output="4", expected="4")
        assert result.score == 0.6
        assert result.passed
        assert len(judge._llm.calls) == 1

    def test_garbage_then_json_retries_once_and_succeeds(self):
        judge = _judge(["I refuse.", json.dumps({"score": 0.8, "reasoning": "fine"})])
        result = judge.score(input="2+2?", output="4", expected="4")
        assert result.score == 0.8
        assert result.passed
        assert len(judge._llm.calls) == 2

        retry_messages = judge._llm.calls[1]["messages"]
        assert retry_messages[-2].content == "I refuse."  # its own reply, handed back
        assert retry_messages[-1].content == _RETRY_INSTRUCTION

    def test_json_without_score_retries(self):
        judge = _judge(
            [
                json.dumps({"reasoning": "no score field"}),
                json.dumps({"score": 0.7, "reasoning": "second try"}),
            ]
        )
        result = judge.score(input="2+2?", output="4", expected="4")
        assert result.score == 0.7
        assert len(judge._llm.calls) == 2

    def test_garbage_twice_fails_loudly(self):
        judge = _judge(["I refuse.", "Still refusing."])
        result = judge.score(input="2+2?", output="4", expected="4")
        assert result.score == 0.0
        assert not result.passed
        assert "unparseable" in result.reason
        assert "Still refusing." in result.reason  # the raw reply, for diagnosis
        assert len(judge._llm.calls) == 2

    def test_transport_error_is_distinct_and_not_retried(self):
        calls = []

        def boom(messages):
            calls.append(messages)
            raise RuntimeError("connection reset")

        judge = LLMJudge(llm=FunctionModel(boom))
        result = judge.score(input="2+2?", output="4", expected="4")
        assert result.score == 0.0
        assert not result.passed
        assert "connection reset" in result.reason
        assert "unparseable" not in result.reason
        assert len(calls) == 1


class TestJudgeTemplateIntegration:
    def test_legacy_alias_template_interpolates(self):
        """The Phase 0 bug: a platform template using {expected_output} reached the
        judge uninterpolated, so it never saw the expected answer."""
        judge = _judge(
            json.dumps({"score": 1.0, "reasoning": "ok"}),
            prompt_template="Q: {input}\nExpected: {expected_output}\nGot: {{output}}",
        )
        judge.score(input="capital of Japan?", output="Tokyo", expected="Tokyo")

        prompt = judge._llm.calls[0]["messages"][-1].content
        assert prompt == "Q: capital of Japan?\nExpected: Tokyo\nGot: Tokyo"

    def test_missing_expected_renders_na(self):
        judge = _judge(json.dumps({"score": 1.0, "reasoning": "ok"}))
        judge.score(input="hi", output="hello")
        assert "N/A" in judge._llm.calls[0]["messages"][-1].content


class TestScaleAndThreshold:
    def test_scale_is_normalized_to_unit(self):
        judge = _judge(json.dumps({"score": 4, "reasoning": "good"}), scale="1-5")
        assert judge.score(input="q", output="a", expected="e").score == 0.75

    def test_threshold_is_honoured(self):
        judge = _judge(json.dumps({"score": 0.6, "reasoning": "meh"}), threshold=0.8)
        result = judge.score(input="q", output="a", expected="e")
        assert result.score == 0.6
        assert not result.passed

    def test_default_judge_behaviour_unchanged(self):
        """Regression guard: the default (binary scale, 0.5 threshold) path is untouched."""
        judge = _judge(json.dumps({"score": 0.8, "reasoning": "good"}))
        result = judge.score(input="q", output="a", expected="e")
        assert result.score == 0.8
        assert result.passed


class TestTemplateWarning:
    def test_placeholder_free_template_warns(self):
        with pytest.warns(UserWarning, match="no known placeholder"):
            LLMJudge(prompt_template="Just decide if it is good.")

    def test_template_with_alias_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            LLMJudge(prompt_template="Compare {output} to {expected_output}.")

    def test_default_template_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            LLMJudge(criteria="correctness")


class TestGEvalRobustness:
    """G-Eval had the identical strict-parse bug."""

    def test_prose_reply_uses_regex_fallback(self):
        model = TestModel(response='After review, "score": 4, "reasoning": "mostly right"')
        judge = GEval(evaluation_steps=["Compare to expected."], scale="1-5", llm=model)
        result = judge.score(input="q", output="a", expected="e")
        assert result.score == 0.75  # regex fallback AND scale normalization applied
        assert result.passed

    def test_garbage_then_json_retries(self):
        model = TestModel(response=["no.", json.dumps({"score": 5, "reasoning": "ok"})])
        judge = GEval(evaluation_steps=["Compare to expected."], scale="1-5", llm=model)
        result = judge.score(input="q", output="a", expected="e")
        assert result.score == 1.0
        assert len(model.calls) == 2

    def test_garbage_twice_fails_loudly(self):
        model = TestModel(response=["no.", "still no."])
        judge = GEval(evaluation_steps=["Compare to expected."], scale="1-5", llm=model)
        result = judge.score(input="q", output="a", expected="e")
        assert result.score == 0.0
        assert "unparseable" in result.reason
