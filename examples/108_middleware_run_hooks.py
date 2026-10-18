"""Example 108: middleware around the whole run, and around every model call (1.91.0).

A support agent that looks up orders, with five cross-cutting concerns written
once as middleware instead of being wired into the agent:

    TagTicket          before_run      put the ticket id in front of the request
    TimeAndRetry       wrap_model      time every model call; retry one failure
    CheapFinalAnswer   before_model    let a cheaper model write the final answer
                                       (ctx.llm — one call, not the agent)
    AlertOnFailure     on_error        log why a run failed (it still fails)
    HideUsage          on_stream_event keep token counts out of the stream

``wrap_model`` sees every model call — each turn of the tool loop and the
structured-output / guardrail re-asks. On ``astream`` the model's deltas still
reach you as they arrive.

Usage:
    python examples/108_middleware_run_hooks.py --offline     # no keys needed
    zsh -lc 'python examples/108_middleware_run_hooks.py'      # OPENAI_API_KEY (+ GEMINI_API_KEY)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

from fastaiagent import Agent, AgentMiddleware, FunctionTool, LLMClient
from fastaiagent.llm.stream import TextDelta, Usage

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def get_order_status(order_id: str) -> str:
    """Current status of an order."""
    return {"A-17": "shipped on 2026-10-14, arriving 2026-10-20"}.get(order_id, "unknown order")


class TagTicket(AgentMiddleware):
    async def before_run(self, ctx, input):
        return f"[ticket T-881] {input}"


class TimeAndRetry(AgentMiddleware):
    async def wrap_model(self, ctx, messages, call_next):
        started = time.monotonic()
        try:
            return await call_next(messages)
        except Exception as exc:  # one retry, then let it fail
            logging.info("model call failed (%s) — retrying once", type(exc).__name__)
            return await call_next(messages)
        finally:
            ms = int((time.monotonic() - started) * 1000)
            ctx.scratch.setdefault("model_ms", []).append((ctx.llm.model, ms))

    async def after_run(self, ctx, result):
        for model, ms in ctx.scratch.get("model_ms", []):
            print(f"  model call: {model:<24} {ms:>5} ms")
        print(f"  run spend so far: {ctx.usage.tokens} tokens, cost known={ctx.usage.cost_known}")
        return result


class CheapFinalAnswer(AgentMiddleware):
    def __init__(self, cheap: LLMClient | None):
        self.cheap = cheap

    async def before_model(self, ctx, messages):
        if self.cheap is not None and ctx.turn >= 1:
            ctx.llm = self.cheap  # this call only; the agent keeps its own model
        return messages


class AlertOnFailure(AgentMiddleware):
    async def on_error(self, ctx, error):
        logging.error("run %s failed: %s", ctx.execution_id, type(error).__name__)


class HideUsage(AgentMiddleware):
    async def on_stream_event(self, ctx, event):
        return None if isinstance(event, Usage) else event


def build(offline: bool) -> Agent:
    if offline:
        from fastaiagent.testing import FunctionModel

        def scripted(messages):
            # Turn 0 asks for the tool; once its result is in, answer.
            if any(m.role.value == "tool" for m in messages):
                return "Order A-17 shipped on 2026-10-14 and arrives on 2026-10-20."
            return "", [{"name": "get_order_status", "arguments": {"order_id": "A-17"}}]

        llm: LLMClient = FunctionModel(scripted)
        cheap = None
    else:
        llm = LLMClient(provider="openai", model="gpt-5-mini")
        cheap = (
            LLMClient(provider="gemini", model="gemini-3.5-flash-lite")
            if os.environ.get("GEMINI_API_KEY")
            else None
        )
    return Agent(
        name="support-desk",
        system_prompt="Use the order tool, then answer in one sentence.",
        llm=llm,
        tools=[FunctionTool(name="get_order_status", fn=get_order_status)],
        middleware=[
            TagTicket(),
            TimeAndRetry(),
            CheapFinalAnswer(cheap),
            AlertOnFailure(),
            HideUsage(),
        ],
    )


async def main(offline: bool) -> None:
    print("── run")
    result = await build(offline).arun("Where is order A-17?")
    print(f"  output: {result.output}")

    print("── astream (deltas as they arrive; Usage events dropped by HideUsage)")
    kinds: set[str] = set()
    async for event in build(offline).astream("Where is order A-17?"):
        kinds.add(type(event).__name__)
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
    print(f"\n  event types seen: {sorted(kinds)}")


if __name__ == "__main__":
    offline = "--offline" in sys.argv
    if not offline and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — running --offline")
        offline = True
    asyncio.run(main(offline))
