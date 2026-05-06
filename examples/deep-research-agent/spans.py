"""
Structured trace spans for the deep-research pipeline.

The pipeline emits four span kinds (in addition to the spans the SDK creates
for LLM calls and tool calls automatically):

  * ``deep_research.session``  — root span for one research run
  * ``deep_research.scope``    — scope phase (brief generation)
  * ``deep_research.research`` — single sub-researcher branch
  * ``deep_research.write``    — write phase (one-shot report)

Plan, brief, and findings are persisted as **structured JSON in span
attributes** so the local UI / replay tooling can reconstruct them. We use
the ``fastaiagent.research.*`` namespace — these keys are also registered
in ``fastaiagent/trace/span.py`` for discoverability.

This module is intentionally tiny: it just wraps ``span.set_attribute``
with a typed surface. No SDK changes required.
"""

from __future__ import annotations

import json
from typing import Any

# Attribute key constants — keep aligned with fastaiagent/trace/span.py
ATTR_BRIEF = "fastaiagent.research.brief"
ATTR_PLAN = "fastaiagent.research.plan"
ATTR_FINDINGS = "fastaiagent.research.findings"
ATTR_TOPIC = "fastaiagent.research.topic"
ATTR_SUBTOPIC = "fastaiagent.research.subtopic"
ATTR_REPORT_LEN = "fastaiagent.research.report.chars"
ATTR_REPORT_CITATIONS = "fastaiagent.research.report.citations"


def _json(value: Any) -> str:
    """Serialize for span attribute storage. Falls back to ``str`` on errors."""
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        return str(value)


def set_topic(span: Any, topic: str) -> None:
    """Tag the session span with the user-provided research topic."""
    span.set_attribute(ATTR_TOPIC, topic)


def set_brief(span: Any, brief: Any) -> None:
    """Persist the scope-phase research brief on the scope span.

    ``brief`` should be a Pydantic model or dict. We store the JSON form so
    the UI can render it without re-parsing prose.
    """
    payload = brief.model_dump() if hasattr(brief, "model_dump") else brief
    span.set_attribute(ATTR_BRIEF, _json(payload))


def set_plan(span: Any, plan: Any) -> None:
    """Persist the research plan (subtopic list) on the session span."""
    payload = plan.model_dump() if hasattr(plan, "model_dump") else plan
    span.set_attribute(ATTR_PLAN, _json(payload))


def set_subtopic(span: Any, subtopic: str) -> None:
    """Tag a research-branch span with the subtopic it covered."""
    span.set_attribute(ATTR_SUBTOPIC, subtopic)


def set_findings(span: Any, findings: Any) -> None:
    """Persist the structured findings of a research branch."""
    payload = findings.model_dump() if hasattr(findings, "model_dump") else findings
    span.set_attribute(ATTR_FINDINGS, _json(payload))


def set_report_metadata(span: Any, report: str) -> None:
    """Lightweight metadata about the final report on the write span."""
    span.set_attribute(ATTR_REPORT_LEN, len(report))
    # Count citation markers like [1], [2], … as a cheap quality signal.
    import re

    citations = len(re.findall(r"\[\d+\]", report))
    span.set_attribute(ATTR_REPORT_CITATIONS, citations)
