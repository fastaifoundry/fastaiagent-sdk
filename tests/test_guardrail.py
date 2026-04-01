"""Tests for fastaiagent.guardrail module."""

from __future__ import annotations

import pytest

from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail import (
    Guardrail,
    GuardrailPosition,
    GuardrailResult,
    GuardrailType,
    allowed_domains,
    execute_guardrails,
    json_valid,
    no_pii,
    toxicity_check,
)

# --- Guardrail base tests ---


class TestGuardrail:
    def test_construction(self):
        g = Guardrail(name="test", guardrail_type=GuardrailType.code)
        assert g.name == "test"
        assert g.guardrail_type == GuardrailType.code
        assert g.blocking is True

    def test_to_dict(self):
        g = Guardrail(
            name="test",
            guardrail_type=GuardrailType.regex,
            position=GuardrailPosition.input,
            config={"pattern": "\\d+"},
            blocking=False,
        )
        d = g.to_dict()
        assert d["name"] == "test"
        assert d["guardrail_type"] == "regex"
        assert d["position"] == "input"
        assert d["blocking"] is False

    def test_from_dict(self):
        data = {
            "name": "test",
            "guardrail_type": "llm_judge",
            "position": "output",
            "config": {"prompt": "Is this good?"},
            "blocking": True,
        }
        g = Guardrail.from_dict(data)
        assert g.name == "test"
        assert g.guardrail_type == GuardrailType.llm_judge
        assert g.position == GuardrailPosition.output

    def test_roundtrip(self):
        original = Guardrail(
            name="test",
            guardrail_type=GuardrailType.schema,
            position=GuardrailPosition.tool_result,
            config={"schema": {"type": "object"}},
        )
        restored = Guardrail.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.guardrail_type == original.guardrail_type
        assert restored.position == original.position


# --- Code guardrail tests ---


class TestCodeGuardrail:
    @pytest.mark.asyncio
    async def test_inline_function(self):
        g = Guardrail(
            name="length_check",
            fn=lambda text: len(text) < 100,
        )
        result = await g.aexecute("short text")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_inline_function_fails(self):
        g = Guardrail(
            name="length_check",
            fn=lambda text: len(text) < 5,
        )
        result = await g.aexecute("this is too long")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_function_returning_guardrail_result(self):
        def check(text):
            return GuardrailResult(passed=True, score=0.95, message="All good")

        g = Guardrail(name="custom", fn=check)
        result = await g.aexecute("test")
        assert result.passed is True
        assert result.score == 0.95


# --- Regex guardrail tests ---


class TestRegexGuardrail:
    @pytest.mark.asyncio
    async def test_pattern_not_found_passes(self):
        g = Guardrail(
            name="no_emails",
            guardrail_type=GuardrailType.regex,
            config={
                "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                "should_match": False,
            },
        )
        result = await g.aexecute("Hello world")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_pattern_found_blocks(self):
        g = Guardrail(
            name="no_emails",
            guardrail_type=GuardrailType.regex,
            config={
                "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                "should_match": False,
            },
        )
        result = await g.aexecute("Contact me at user@example.com")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_should_match_passes(self):
        g = Guardrail(
            name="has_number",
            guardrail_type=GuardrailType.regex,
            config={"pattern": r"\d+", "should_match": True},
        )
        result = await g.aexecute("Order #12345")
        assert result.passed is True


# --- Schema guardrail tests ---


class TestSchemaGuardrail:
    @pytest.mark.asyncio
    async def test_valid_schema(self):
        g = Guardrail(
            name="schema_check",
            guardrail_type=GuardrailType.schema,
            config={
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
        )
        result = await g.aexecute({"name": "Alice"})
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_invalid_schema(self):
        g = Guardrail(
            name="schema_check",
            guardrail_type=GuardrailType.schema,
            config={
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
        )
        result = await g.aexecute({"age": 30})
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_invalid_json_string(self):
        g = Guardrail(
            name="schema_check",
            guardrail_type=GuardrailType.schema,
            config={"schema": {"type": "object"}},
        )
        result = await g.aexecute("not json {{{")
        assert result.passed is False


# --- Classifier guardrail tests ---


class TestClassifierGuardrail:
    @pytest.mark.asyncio
    async def test_no_blocked_categories(self):
        g = Guardrail(
            name="classifier",
            guardrail_type=GuardrailType.classifier,
            config={
                "categories": {"finance": ["money", "bank"], "tech": ["code", "api"]},
                "blocked": ["finance"],
            },
        )
        result = await g.aexecute("Write some code for the api")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_blocked_category(self):
        g = Guardrail(
            name="classifier",
            guardrail_type=GuardrailType.classifier,
            config={
                "categories": {"finance": ["money", "bank"]},
                "blocked": ["finance"],
            },
        )
        result = await g.aexecute("Send money to the bank")
        assert result.passed is False


# --- Built-in factory tests ---


class TestBuiltins:
    @pytest.mark.asyncio
    async def test_no_pii_passes(self):
        g = no_pii()
        result = await g.aexecute("Hello World")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_no_pii_detects_ssn(self):
        g = no_pii()
        result = await g.aexecute("SSN: 123-45-6789")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_no_pii_detects_email(self):
        g = no_pii()
        result = await g.aexecute("Contact: user@example.com")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_json_valid_passes(self):
        g = json_valid()
        result = await g.aexecute('{"key": "value"}')
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_json_valid_fails(self):
        g = json_valid()
        result = await g.aexecute("not json")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_toxicity_passes(self):
        g = toxicity_check()
        result = await g.aexecute("Hello, how can I help?")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_toxicity_detects(self):
        g = toxicity_check()
        result = await g.aexecute("I will attack you")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_allowed_domains_passes(self):
        g = allowed_domains(["example.com", "api.test.com"])
        result = await g.aexecute("Visit https://api.test.com/data")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_allowed_domains_blocks(self):
        g = allowed_domains(["example.com"])
        result = await g.aexecute("Visit https://evil.com/phishing")
        assert result.passed is False


# --- Executor tests ---


class TestExecutor:
    @pytest.mark.asyncio
    async def test_execute_guardrails_all_pass(self):
        guardrails = [
            Guardrail(
                name="check1",
                position=GuardrailPosition.output,
                fn=lambda text: True,
            ),
            Guardrail(
                name="check2",
                position=GuardrailPosition.output,
                fn=lambda text: True,
            ),
        ]
        results = await execute_guardrails(
            guardrails, "test data", GuardrailPosition.output
        )
        assert len(results) == 2
        assert all(r.passed for r in results)

    @pytest.mark.asyncio
    async def test_blocking_guardrail_raises(self):
        guardrails = [
            Guardrail(
                name="blocker",
                position=GuardrailPosition.output,
                blocking=True,
                fn=lambda text: False,
            ),
        ]
        with pytest.raises(GuardrailBlockedError, match="blocker"):
            await execute_guardrails(
                guardrails, "test data", GuardrailPosition.output
            )

    @pytest.mark.asyncio
    async def test_non_blocking_doesnt_raise(self):
        guardrails = [
            Guardrail(
                name="non_blocker",
                position=GuardrailPosition.output,
                blocking=False,
                fn=lambda text: False,
            ),
        ]
        results = await execute_guardrails(
            guardrails, "test data", GuardrailPosition.output
        )
        assert len(results) == 1
        assert results[0].passed is False

    @pytest.mark.asyncio
    async def test_filters_by_position(self):
        guardrails = [
            Guardrail(
                name="input_guard",
                position=GuardrailPosition.input,
                fn=lambda text: True,
            ),
            Guardrail(
                name="output_guard",
                position=GuardrailPosition.output,
                fn=lambda text: True,
            ),
        ]
        results = await execute_guardrails(
            guardrails, "test", GuardrailPosition.input
        )
        assert len(results) == 1
