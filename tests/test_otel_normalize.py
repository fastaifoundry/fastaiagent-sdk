"""Unit tests for the foreign-span attribute normalizer.

Pure-function tests — no DB, no provider, no mocks. They assert that
``normalize_attributes`` maps OpenInference and OpenLLMetry conventions onto the
exact canonical keys the rest of the stack reads, never overwrites existing
keys, and preserves originals + unknown keys.
"""

from __future__ import annotations

from fastaiagent.trace.normalize import normalize_attributes


class TestOpenInference:
    def test_maps_core_keys(self) -> None:
        raw = {
            "llm.model_name": "gpt-4o-mini",
            "llm.token_count.prompt": 12,
            "llm.token_count.completion": 8,
            "input.value": "Hello",
            "output.value": "Hi there",
            "openinference.span.kind": "LLM",
            "llm.system": "openai",
            "custom.unknown": "keep-me",
        }
        out = normalize_attributes(
            raw,
            scope_name="openinference.instrumentation.openai",
            is_root=True,
        )

        assert out["gen_ai.request.model"] == "gpt-4o-mini"
        assert out["gen_ai.usage.input_tokens"] == 12
        assert out["gen_ai.usage.output_tokens"] == 8
        # Prompt/completion fan out to BOTH the FTS keys and the UI IO-panel keys.
        assert out["gen_ai.prompt"] == "Hello"  # FTS / search
        assert out["gen_ai.completion"] == "Hi there"  # FTS / search
        assert out["gen_ai.request.messages"] == "Hello"  # UI input panel
        assert out["gen_ai.response.content"] == "Hi there"  # UI output panel
        assert out["gen_ai.system"] == "openai"
        assert out["fastaiagent.runner.type"] == "llm"
        assert out["fastaiagent.framework"] == "openai"

        # Originals preserved, unknown keys pass through untouched.
        assert out["llm.model_name"] == "gpt-4o-mini"
        assert out["input.value"] == "Hello"
        assert out["custom.unknown"] == "keep-me"

    def test_span_kind_mapping(self) -> None:
        for kind, expected in [
            ("CHAIN", "chain"),
            ("AGENT", "agent"),
            ("TOOL", "tool"),
            ("RETRIEVER", "retrieval"),
            ("EMBEDDING", "embedding"),
        ]:
            out = normalize_attributes({"openinference.span.kind": kind})
            assert out["fastaiagent.runner.type"] == expected

    def test_tool_name_implies_tool_runner(self) -> None:
        out = normalize_attributes({"tool.name": "search"})
        assert out["fastaiagent.tool.name"] == "search"
        assert out["fastaiagent.runner.type"] == "tool"

    def test_invocation_parameters_json(self) -> None:
        out = normalize_attributes(
            {"llm.invocation_parameters": '{"temperature": 0.7, "max_tokens": 256}'}
        )
        assert out["gen_ai.request.temperature"] == 0.7
        assert out["gen_ai.request.max_tokens"] == 256

    def test_invocation_parameters_invalid_json_is_ignored(self) -> None:
        out = normalize_attributes({"llm.invocation_parameters": "not-json"})
        assert "gen_ai.request.temperature" not in out


class TestOpenLLMetry:
    def test_legacy_model_and_token_spellings(self) -> None:
        raw = {
            "llm.request.model": "claude-3-5-haiku",
            "gen_ai.usage.prompt_tokens": 30,
            "gen_ai.usage.completion_tokens": 15,
        }
        out = normalize_attributes(raw)
        assert out["gen_ai.request.model"] == "claude-3-5-haiku"
        assert out["gen_ai.usage.input_tokens"] == 30
        assert out["gen_ai.usage.output_tokens"] == 15

    def test_indexed_messages_are_consolidated(self) -> None:
        raw = {
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "Question one",
            "gen_ai.prompt.1.content": "Question two",
            "gen_ai.completion.0.content": "Answer",
        }
        out = normalize_attributes(raw)
        assert out["gen_ai.prompt"] == "Question one\nQuestion two"
        assert out["gen_ai.completion"] == "Answer"
        # Same fan-out to the UI IO-panel keys.
        assert out["gen_ai.request.messages"] == "Question one\nQuestion two"
        assert out["gen_ai.response.content"] == "Answer"


class TestNonDestructive:
    def test_existing_canonical_keys_not_overwritten(self) -> None:
        raw = {
            "gen_ai.request.model": "gpt-4o",  # already canonical
            "llm.model_name": "should-not-win",  # foreign, must not overwrite
            "gen_ai.usage.input_tokens": 100,
        }
        out = normalize_attributes(raw)
        assert out["gen_ai.request.model"] == "gpt-4o"
        assert out["gen_ai.usage.input_tokens"] == 100

    def test_native_span_is_noop(self) -> None:
        # A canonical native span with no foreign keys must come back unchanged.
        native = {
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.usage.input_tokens": 5,
            "gen_ai.usage.output_tokens": 3,
            "fastaiagent.runner.type": "agent",
            "fastaiagent.framework": "fastaiagent",
        }
        out = normalize_attributes(
            native, scope_name="fastaiagent", is_root=True
        )
        assert out == native

    def test_framework_only_on_root(self) -> None:
        raw = {"llm.model_name": "gpt-4o-mini"}
        non_root = normalize_attributes(
            raw, scope_name="openinference.instrumentation.openai", is_root=False
        )
        assert "fastaiagent.framework" not in non_root

        root = normalize_attributes(
            raw, scope_name="openinference.instrumentation.openai", is_root=True
        )
        assert root["fastaiagent.framework"] == "openai"

    def test_framework_override_wins(self) -> None:
        out = normalize_attributes(
            {"llm.model_name": "x"},
            scope_name="openinference.instrumentation.openai",
            is_root=True,
            framework_override="my-stack",
        )
        assert out["fastaiagent.framework"] == "my-stack"

    def test_input_is_not_mutated(self) -> None:
        raw = {"llm.model_name": "gpt-4o-mini"}
        before = dict(raw)
        normalize_attributes(raw, is_root=True, scope_name="x.openai")
        assert raw == before  # caller's dict untouched


class TestFrameworkFromScope:
    def test_known_frameworks(self) -> None:
        cases = {
            "openinference.instrumentation.openai": "openai",
            "openinference.instrumentation.langchain": "langchain",
            "opentelemetry.instrumentation.anthropic": "anthropic",
            "opentelemetry.instrumentation.crewai": "crewai",
        }
        for scope, expected in cases.items():
            out = normalize_attributes({}, scope_name=scope, is_root=True)
            assert out.get("fastaiagent.framework") == expected

    def test_unknown_scope_falls_back_to_last_segment(self) -> None:
        out = normalize_attributes(
            {}, scope_name="vendor.instrumentation.acme", is_root=True
        )
        assert out["fastaiagent.framework"] == "acme"
