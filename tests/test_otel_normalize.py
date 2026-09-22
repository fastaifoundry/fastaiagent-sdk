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

    def test_native_span_keys_are_never_rewritten(self) -> None:
        """A canonical span keeps every key and value it arrived with.

        This used to assert ``out == native`` — a strict no-op. The contract
        narrowed deliberately when the normalizer began pricing spans: it may
        now *add* ``fastaiagent.cost.total_usd`` when the span carries a model
        and tokens but no cost, because a captured foreign span never had one
        and the control plane has no read-time pricing step to supply it.

        Nothing is rewritten. The span below is synthetic — a real native span
        with a priceable model already carries its cost from the call site, so
        the guard skips it, and one without a price stays without a price.
        """
        native = {
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.usage.input_tokens": 5,
            "gen_ai.usage.output_tokens": 3,
            "fastaiagent.runner.type": "agent",
            "fastaiagent.framework": "fastaiagent",
        }
        out = normalize_attributes(native, scope_name="fastaiagent", is_root=True)

        for key, value in native.items():
            assert out[key] == value, f"{key} was rewritten"
        assert set(out) - set(native) <= {"fastaiagent.cost.total_usd"}

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
        out = normalize_attributes({}, scope_name="vendor.instrumentation.acme", is_root=True)
        assert out["fastaiagent.framework"] == "acme"


class TestForeignSpanCost:
    """A captured foreign span must carry its own cost, when we can price it.

    Before this, the normalizer mapped the model and the token counts and
    stopped: a foreign span reached storage with tokens and no
    ``fastaiagent.cost.total_usd``, and the Local UI priced it at read time.
    The control plane has no read-time step, so it priced those spans from its
    own table — and that is where it went wrong. ``gpt-4.1`` was missing from
    it, a fuzzy prefix match charged the 2023 ``gpt-4`` rate instead, and 187
    live spans were over-reported by 10.9x.

    The plane now prefers the SDK's figure whenever it is present, so every
    span that carries the attribute is one no downstream table has to guess at.
    """

    def test_openinference_span_is_priced(self) -> None:
        out = normalize_attributes(
            {
                "llm.model_name": "gpt-4.1",
                "llm.token_count.prompt": 1000,
                "llm.token_count.completion": 1000,
                "llm.system": "openai",
            }
        )

        assert out["fastaiagent.cost.total_usd"] == 0.01

    def test_openllmetry_span_is_priced(self) -> None:
        out = normalize_attributes(
            {
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.usage.prompt_tokens": 1000,
                "gen_ai.usage.completion_tokens": 1000,
                "gen_ai.system": "openai",
            }
        )

        assert out["fastaiagent.cost.total_usd"] > 0

    def test_an_existing_cost_is_never_overwritten(self) -> None:
        """A span priced at the call keeps that figure — it is the better one."""
        out = normalize_attributes(
            {
                "gen_ai.request.model": "gpt-4.1",
                "gen_ai.usage.input_tokens": 1000,
                "gen_ai.usage.output_tokens": 1000,
                "fastaiagent.cost.total_usd": 0.42,
            }
        )

        assert out["fastaiagent.cost.total_usd"] == 0.42

    def test_an_unpriceable_model_stays_absent(self) -> None:
        """Absent means "we could not price this". It must not become 0.0.

        A fabricated zero is worse than no value: it reads as free, and it
        denies the reader the chance to fall back to their own estimate.
        """
        out = normalize_attributes(
            {
                "gen_ai.request.model": "my-private-finetune",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 10,
            }
        )

        assert "fastaiagent.cost.total_usd" not in out

    def test_a_model_we_do_not_stock_is_not_fuzzy_matched(self) -> None:
        """The exact trap that cost the plane 10.9x: do not guess a rate.

        Bare ``gpt-4`` has no entry here. A prefix matcher would happily answer
        with a neighbouring tier; refusing is the whole point.
        """
        out = normalize_attributes(
            {
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.input_tokens": 1000,
                "gen_ai.usage.output_tokens": 1000,
            }
        )

        assert "fastaiagent.cost.total_usd" not in out

    def test_a_self_hosted_model_is_a_known_zero(self) -> None:
        """Free is not the same as unpriced, and the attribute says which."""
        out = normalize_attributes(
            {
                "gen_ai.request.model": "llama3",
                "gen_ai.usage.input_tokens": 1000,
                "gen_ai.usage.output_tokens": 1000,
                "gen_ai.system": "ollama",
            }
        )

        assert out["fastaiagent.cost.total_usd"] == 0.0

    def test_missing_tokens_or_model_price_nothing(self) -> None:
        assert "fastaiagent.cost.total_usd" not in normalize_attributes(
            {"gen_ai.request.model": "gpt-4.1"}
        )
        assert "fastaiagent.cost.total_usd" not in normalize_attributes(
            {"gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 10}
        )

    def test_normalizing_is_still_pure(self) -> None:
        """Pricing is a table lookup; the input dict must not be touched."""
        raw = {
            "llm.model_name": "gpt-4.1",
            "llm.token_count.prompt": 1000,
            "llm.token_count.completion": 1000,
        }
        before = dict(raw)

        normalize_attributes(raw)

        assert raw == before
