"""security_audit_2 N13 — regex guardrails are ReDoS-bounded and fail closed.

Regex guardrails run on the ``regex`` engine under a hard ``timeout=`` (stdlib
``re`` can't be time-limited and would freeze the whole process on a
catastrophic-backtracking input). A pattern that overruns yields a FAIL, not a
hang. No mocking — real matching against real inputs.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from fastaiagent.guardrail import implementations as impl


def _gr(**config):
    return SimpleNamespace(config=config)


@pytest.mark.asyncio
async def test_normal_match_and_nonmatch():
    assert (await impl._run_regex(_gr(pattern="hello", should_match=True), "hi hello")).passed
    assert (await impl._run_regex(_gr(pattern="nope", should_match=False), "clean text")).passed
    # should_match semantics: present but should_match=False -> FAIL
    assert not (await impl._run_regex(_gr(pattern="bad", should_match=False), "so bad")).passed


@pytest.mark.asyncio
async def test_invalid_regex_fails_closed():
    r = await impl._run_regex(_gr(pattern="(", should_match=True), "x")
    assert not r.passed
    assert "Invalid regex" in r.message


@pytest.mark.asyncio
async def test_redos_pattern_fails_closed_within_timeout():
    # Catastrophic for the regex engine; must abort at the 2s default, not hang.
    start = time.monotonic()
    r = await impl._run_regex(_gr(pattern=r"(a|a|a)*$", should_match=True), "a" * 60 + "b")
    elapsed = time.monotonic() - start
    assert not r.passed
    assert "exceeded" in r.message
    assert elapsed < 4.0, f"guardrail did not abort promptly: {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_custom_timeout_is_honored_and_clamped():
    # A short custom timeout aborts sooner.
    start = time.monotonic()
    r = await impl._run_regex(
        _gr(pattern=r"(a|a|a)*$", should_match=True, timeout_seconds=0.5),
        "a" * 60 + "b",
    )
    assert not r.passed and (time.monotonic() - start) < 1.5

    # A plane-supplied huge value can't disable the protection.
    assert impl._resolve_regex_timeout({"timeout_seconds": 3600}) == 10.0
    assert impl._resolve_regex_timeout({"timeout_seconds": 0.001}) == 0.1
    assert impl._resolve_regex_timeout({"timeout_seconds": "garbage"}) == 2.0
    assert impl._resolve_regex_timeout({}) == 2.0
