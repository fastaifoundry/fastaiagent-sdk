"""Guardrail implementation runners for each type."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from fastaiagent.guardrail.guardrail import GuardrailResult, GuardrailType

if TYPE_CHECKING:
    from fastaiagent.guardrail.guardrail import Guardrail


async def run_guardrail(guardrail: Guardrail, data: str | dict) -> GuardrailResult:
    """Run a guardrail based on its implementation type."""
    runners = {
        GuardrailType.code: _run_code,
        GuardrailType.llm_judge: _run_llm_judge,
        GuardrailType.regex: _run_regex,
        GuardrailType.schema: _run_schema,
        GuardrailType.classifier: _run_classifier,
    }
    runner = runners.get(guardrail.guardrail_type, _run_code)
    return await runner(guardrail, data)


async def _run_code(guardrail: Guardrail, data: str | dict) -> GuardrailResult:
    """Execute a code guardrail — runs a Python function."""
    if guardrail.fn is not None:
        try:
            text = data if isinstance(data, str) else json.dumps(data)
            result = guardrail.fn(text)
            if isinstance(result, bool):
                return GuardrailResult(passed=result)
            if isinstance(result, GuardrailResult):
                return result
            return GuardrailResult(passed=bool(result))
        except Exception as e:
            return GuardrailResult(passed=False, message=f"Code guardrail error: {e}")

    # Config-based code execution (sandboxed)
    code = guardrail.config.get("code", "")
    if not code:
        return GuardrailResult(passed=True, message="No code configured")

    text = data if isinstance(data, str) else json.dumps(data)
    safe_globals: dict[str, Any] = {
        "__builtins__": {
            "len": len, "str": str, "int": int, "float": float,
            "bool": bool, "list": list, "dict": dict,
            "re": re, "json": json,
        }
    }
    safe_locals: dict[str, Any] = {"data": text, "result": True}

    try:
        exec(code, safe_globals, safe_locals)  # noqa: S102
        return GuardrailResult(passed=bool(safe_locals.get("result", True)))
    except Exception as e:
        return GuardrailResult(passed=False, message=f"Code execution error: {e}")


async def _run_llm_judge(guardrail: Guardrail, data: str | dict) -> GuardrailResult:
    """Execute an LLM judge guardrail."""
    from fastaiagent.llm import LLMClient, SystemMessage, UserMessage

    prompt_template = guardrail.config.get(
        "prompt", "Evaluate if the following is acceptable. Respond with PASS or FAIL.\n\n{data}"
    )
    pass_value = guardrail.config.get("pass_value", "PASS")

    text = data if isinstance(data, str) else json.dumps(data)
    prompt = prompt_template.replace("{data}", text)

    llm_config = guardrail.config.get("llm", {})
    llm = LLMClient(**llm_config) if llm_config else LLMClient()

    try:
        response = await llm.acomplete([
            SystemMessage("You are a judge. Respond with PASS or FAIL only."),
            UserMessage(prompt),
        ])
        content = (response.content or "").strip().upper()
        passed = pass_value.upper() in content
        return GuardrailResult(passed=passed, message=response.content)
    except Exception as e:
        return GuardrailResult(passed=False, message=f"LLM judge error: {e}")


async def _run_regex(guardrail: Guardrail, data: str | dict) -> GuardrailResult:
    """Execute a regex guardrail."""
    pattern = guardrail.config.get("pattern", "")
    should_match = guardrail.config.get("should_match", False)

    text = data if isinstance(data, str) else json.dumps(data)

    try:
        flags = 0
        if guardrail.config.get("case_insensitive", False):
            flags |= re.IGNORECASE

        match = re.search(pattern, text, flags)
        if should_match:
            passed = match is not None
        else:
            passed = match is None

        return GuardrailResult(
            passed=passed,
            message=f"Pattern {'matched' if match else 'not matched'}: {pattern}",
        )
    except re.error as e:
        return GuardrailResult(passed=False, message=f"Invalid regex: {e}")


async def _run_schema(guardrail: Guardrail, data: str | dict) -> GuardrailResult:
    """Execute a JSON schema validation guardrail."""
    schema = guardrail.config.get("schema", {})

    try:
        if isinstance(data, str):
            parsed = json.loads(data)
        else:
            parsed = data
    except json.JSONDecodeError as e:
        return GuardrailResult(passed=False, message=f"Invalid JSON: {e}")

    from fastaiagent.tool.schema import validate_schema

    violations = validate_schema(schema, parsed)
    if violations:
        messages = [v.message for v in violations[:3]]
        return GuardrailResult(
            passed=False,
            message=f"Schema violations: {'; '.join(messages)}",
            metadata={"violations": [v.model_dump() for v in violations]},
        )
    return GuardrailResult(passed=True)


async def _run_classifier(guardrail: Guardrail, data: str | dict) -> GuardrailResult:
    """Execute a classifier guardrail (keyword/pattern-based)."""
    categories = guardrail.config.get("categories", {})
    blocked_categories = guardrail.config.get("blocked", [])

    text = data if isinstance(data, str) else json.dumps(data)
    text_lower = text.lower()

    detected = []
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                detected.append(category)
                break

    blocked = [cat for cat in detected if cat in blocked_categories]
    passed = len(blocked) == 0

    return GuardrailResult(
        passed=passed,
        message=(
            f"Detected categories: {detected}. Blocked: {blocked}"
            if detected
            else "No categories detected"
        ),
        metadata={"detected": detected, "blocked": blocked},
    )
