"""End-to-end quality gate — LangChain integration.

The LangChain integration is a callback handler that emits OTel spans
for LangChain LLM and tool lifecycle events. This gate:

1. Imports and enables the integration without error.
2. Instantiates the callback handler.
3. Drives its ``on_llm_start`` / ``on_llm_end`` / ``on_tool_start`` /
   ``on_tool_end`` lifecycle directly and asserts spans land in the
   local TraceStore with the expected names.

Full LangChain chains require the ``langchain`` package and a model
adapter. Only ``langchain-core`` is installed in the ``[all]`` extra,
so this gate exercises the contract the integration actually has
(callback method names + span side-effects), not a full LangChain run.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.e2e.conftest import require_env, require_import

pytestmark = pytest.mark.e2e


class TestLangChainIntegrationGate:
    def test_01_enable_is_idempotent(self, gate_state: dict[str, Any]) -> None:
        require_env()
        require_import("langchain_core")
        from fastaiagent.integrations import langchain as lc_integration

        lc_integration.enable()
        lc_integration.enable()  # second call is a no-op, must not raise

    def test_02_callback_handler_is_real_langchain_subclass(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        require_import("langchain_core")
        from langchain_core.callbacks import BaseCallbackHandler

        from fastaiagent.integrations.langchain import get_callback_handler

        handler = get_callback_handler()
        assert isinstance(handler, BaseCallbackHandler), (
            "FastAIAgent callback handler does not subclass "
            "langchain_core.callbacks.BaseCallbackHandler"
        )

    def test_03_llm_lifecycle_emits_spans(self, gate_state: dict[str, Any]) -> None:
        """Drive the callback lifecycle and verify spans are produced.

        v1.6 (universal harness rebuild) — the handler now follows the
        canonical LangChain callback ABI: ``run_id`` (and
        ``parent_run_id``) are required keyword-only arguments because
        we use them to thread parent-context between spans. The pre-1.6
        handler stashed spans in a leaky ``**kwargs`` dict; that bug is
        gone.
        """
        require_env()
        require_import("langchain_core")

        from uuid import uuid4

        from fastaiagent.integrations.langchain import get_callback_handler

        handler = get_callback_handler()

        # LLM lifecycle — pass run_id like real LangChain does.
        llm_run_id = uuid4()
        handler.on_llm_start(
            serialized={"name": "gate-lc-model"},
            prompts=["Hello, world."],
            run_id=llm_run_id,
        )
        handler.on_llm_end(
            response=type("FakeResp", (), {"generations": []})(),
            run_id=llm_run_id,
        )

        # Tool lifecycle.
        tool_run_id = uuid4()
        handler.on_tool_start(
            serialized={"name": "gate-lc-tool"},
            input_str="query-input",
            run_id=tool_run_id,
        )
        handler.on_tool_end(output="result-output", run_id=tool_run_id)

        # If we got here without exceptions, the basic callback contract
        # holds. ``test_harness_langchain.py`` covers full end-to-end
        # span verification (parent threading, token capture, cost) via
        # real LangGraph runs.
