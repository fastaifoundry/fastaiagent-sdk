"""Unit + integration tests for opt-in trace redaction.

Covers:
* :class:`RedactionPolicy` regex matching + replacement
* Walking nested dicts/lists/strings
* Capture-mode integration through ``LocalStorageProcessor.on_end``
* Mode gating (``off``, ``capture``, ``read``, ``both``)
* Zero-overhead fast path when no policy is installed
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fastaiagent.trace.redaction import (
    RedactionPolicy,
    _capture_redact,
    _read_redact,
    get_redaction_policy,
    redact_attributes,
    set_redaction_policy,
)


@pytest.fixture(autouse=True)
def _reset_policy():
    """Every test starts with no policy installed; restored after."""
    saved = get_redaction_policy()
    set_redaction_policy(None)
    yield
    set_redaction_policy(saved)


class TestRedactionPolicy:
    def test_redact_string_substitutes_regex_match(self):
        policy = RedactionPolicy(patterns=(r"sk-[A-Za-z0-9]{32,}",))
        out = policy.redact_string("my key sk-abcdefghijklmnopqrstuvwxyz123456 ok")
        assert "sk-" not in out
        assert "[REDACTED]" in out

    def test_custom_replacement_token(self):
        policy = RedactionPolicy(patterns=(r"\d+",), replacement="###")
        assert policy.redact_string("call 911 now") == "call ### now"

    def test_multiple_patterns_compose(self):
        policy = RedactionPolicy(patterns=(r"sk-\w+", r"\d{4}-\d{4}"))
        out = policy.redact_string("sk-abcd or 1234-5678")
        assert "sk-" not in out and "1234-5678" not in out

    def test_compile_happens_once_on_construction(self):
        # Cached compiled patterns are kept on the frozen dataclass via
        # __setattr__ in __post_init__ — confirm the cache exists and
        # has the expected count.
        policy = RedactionPolicy(patterns=(r"\d+", r"[A-Z]+"))
        assert len(policy._compiled) == 2


class TestRedactAttributes:
    def test_redacts_only_sensitive_keys(self):
        policy = RedactionPolicy(patterns=(r"sk-\w+",))
        attrs = {
            "gen_ai.response.content": "leaked sk-xyz here",
            "fastaiagent.agent.name": "sk-xyz",  # not sensitive — should pass
        }
        out = redact_attributes(attrs, policy)
        assert "[REDACTED]" in out["gen_ai.response.content"]
        assert out["fastaiagent.agent.name"] == "sk-xyz"

    def test_walks_nested_dict(self):
        policy = RedactionPolicy(patterns=(r"sk-\w+",))
        attrs = {
            "gen_ai.request.messages": {
                "messages": [{"role": "user", "content": "use sk-secret123"}]
            }
        }
        out = redact_attributes(attrs, policy)
        nested = out["gen_ai.request.messages"]["messages"][0]
        assert "[REDACTED]" in nested["content"]
        assert nested["role"] == "user"  # non-string field untouched

    def test_walks_nested_list(self):
        policy = RedactionPolicy(patterns=(r"secret\d+",))
        attrs = {"tool.input": ["call secret42", "harmless"]}
        out = redact_attributes(attrs, policy)
        assert "[REDACTED]" in out["tool.input"][0]
        assert out["tool.input"][1] == "harmless"

    def test_off_mode_is_noop(self):
        policy = RedactionPolicy(patterns=(r".+",), mode="off")
        attrs = {"gen_ai.response.content": "anything"}
        # ``redact_attributes`` short-circuits on mode == "off".
        assert redact_attributes(attrs, policy) == attrs

    def test_no_policy_installed_is_zero_overhead(self):
        # When no policy is installed, the input dict is returned
        # by reference — no copy, no walk, no allocation.
        attrs = {"gen_ai.response.content": "sk-leak"}
        out = redact_attributes(attrs)
        assert out is attrs

    def test_apply_to_keys_can_be_narrowed(self):
        # User opts to redact only a custom subset.
        policy = RedactionPolicy(patterns=(r"\d+",), apply_to_keys=frozenset({"my.custom.key"}))
        attrs = {
            "my.custom.key": "id=999",
            "gen_ai.response.content": "id=888",  # default-sensitive but not in custom set
        }
        out = redact_attributes(attrs, policy)
        assert "[REDACTED]" in out["my.custom.key"]
        assert out["gen_ai.response.content"] == "id=888"


class TestModeGating:
    def test_capture_mode_redacts_on_capture_hook(self):
        set_redaction_policy(RedactionPolicy(patterns=(r"sk-\w+",), mode="capture"))
        out = _capture_redact({"gen_ai.response.content": "leak sk-abc"})
        assert "[REDACTED]" in out["gen_ai.response.content"]

    def test_capture_mode_noops_read_hook(self):
        set_redaction_policy(RedactionPolicy(patterns=(r"sk-\w+",), mode="capture"))
        # Read hook should NOT redact when mode is "capture" only.
        out = _read_redact({"gen_ai.response.content": "leak sk-abc"})
        assert out["gen_ai.response.content"] == "leak sk-abc"

    def test_read_mode_noops_capture_hook(self):
        set_redaction_policy(RedactionPolicy(patterns=(r"sk-\w+",), mode="read"))
        out = _capture_redact({"gen_ai.response.content": "leak sk-abc"})
        assert out["gen_ai.response.content"] == "leak sk-abc"

    def test_both_mode_redacts_on_both_hooks(self):
        set_redaction_policy(RedactionPolicy(patterns=(r"sk-\w+",), mode="both"))
        for hook in (_capture_redact, _read_redact):
            out = hook({"gen_ai.response.content": "leak sk-abc"})
            assert "[REDACTED]" in out["gen_ai.response.content"]


class _StubSpan:
    """Minimal stand-in for the OTel ``ReadableSpan`` surface that
    ``LocalStorageProcessor.on_end`` reads. Real OTel exporters would
    not accept this, but the storage processor only touches the
    attributes we pass — exact duck-typing of just the fields it reads.
    """

    def __init__(self, attributes: dict[str, Any]):
        self.name = "test.span"
        self.attributes = attributes
        self.events: list[Any] = []
        self.start_time = 1_700_000_000_000_000_000
        self.end_time = 1_700_000_000_000_000_001
        self.parent = None

        class _Ctx:
            trace_id = 0x11111111111111111111111111111111
            span_id = 0x2222222222222222

        self._ctx = _Ctx()

        class _Status:
            class _Code:
                name = "OK"

            status_code = _Code()

        self.status = _Status()

    def get_span_context(self):
        return self._ctx


class TestCaptureIntegration:
    """Confirm that an installed capture-mode policy actually rewrites
    SQLite contents — the contract that protects on-disk traces, and
    everything that reads back out of the local store (UI, ``TraceStore``,
    and the platform drain, which re-sends rows *from* SQLite).

    Note this does not extend to OTel exporters registered via
    ``add_exporter``: they are sibling span processors reading
    ``span.attributes`` off the span itself, which is never mutated.
    """

    def test_capture_mode_redacts_stored_attributes(self, tmp_path: Path):
        from fastaiagent.trace.storage import LocalStorageProcessor

        set_redaction_policy(RedactionPolicy(patterns=(r"sk-\w+",), mode="capture"))
        proc = LocalStorageProcessor(db_path=str(tmp_path / "trace.db"))
        span = _StubSpan({"gen_ai.response.content": "leaked sk-DEADBEEF"})
        proc.on_end(span)

        db = proc._get_db()
        rows = db.fetchall("SELECT attributes FROM spans")
        assert len(rows) == 1
        stored = json.loads(rows[0]["attributes"])
        assert "sk-" not in stored["gen_ai.response.content"]
        assert "[REDACTED]" in stored["gen_ai.response.content"]

    def test_off_mode_leaves_storage_raw(self, tmp_path: Path):
        from fastaiagent.trace.storage import LocalStorageProcessor

        # Default: no policy installed — capture path is a pass-through.
        proc = LocalStorageProcessor(db_path=str(tmp_path / "trace.db"))
        span = _StubSpan({"gen_ai.response.content": "raw sk-DEADBEEF"})
        proc.on_end(span)

        db = proc._get_db()
        rows = db.fetchall("SELECT attributes FROM spans")
        stored = json.loads(rows[0]["attributes"])
        assert stored["gen_ai.response.content"] == "raw sk-DEADBEEF"
