"""Example 111: a client that survives a failing model, and knows its limits (1.91.0).

Three things on ``LLMClient``:

    model_info     context window, max output, reasoning, caching, images, PDF —
                   from the providers' own sources, ``None`` when unknown
    fallbacks      after this client's own retries: a rate limit, a 5xx, a dead
                   connection, a timeout or a context overflow → the next model
    extra_body /   provider options the SDK has no parameter for; never in
    extra_headers  to_dict()

The demo points the primary at a port nothing listens on, so the fallback answers.

Usage:
    zsh -lc 'python examples/111_llm_fallbacks_and_model_info.py'   # OPENAI_API_KEY (+ others)
"""

from __future__ import annotations

import os
import socket

from fastaiagent import Agent, LLMClient
from fastaiagent.llm import UserMessage


def _dead_url() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}/v1"


def show_model_info() -> None:
    print("── model_info")
    for provider, model in [
        ("openai", "gpt-5-mini"),
        ("anthropic", "claude-haiku-4-5"),
        ("gemini", "gemini-3.8-flash"),
        ("openai", "my-private-fine-tune"),
    ]:
        info = LLMClient(provider=provider, model=model, api_key="unused").model_info
        print(
            f"  {model:<22} context={info.context_window} max_out={info.max_output_tokens} "
            f"reasoning={info.supports_reasoning} vision={info.supports_vision}"
        )


def show_fallback() -> None:
    print("── fallbacks")
    backups = [LLMClient(provider="openai", model="gpt-5-mini")]
    if os.environ.get("ANTHROPIC_API_KEY"):
        backups.insert(0, LLMClient(provider="anthropic", model="claude-haiku-4-5"))
    llm = LLMClient(
        provider="openai",
        model="gpt-5-mini",
        api_key="unused",
        base_url=_dead_url(),  # nothing listens here: a connection error
        max_retries=1,
        fallbacks=backups,
    )
    result = Agent(name="resilient", llm=llm).run("Reply with one short sentence about resilience.")
    print(f"  answered: {result.output}")
    print(f"  cost: ${result.cost:.6f} (priced at the model that answered)")


def show_passthrough() -> None:
    print("── extra_body / extra_headers")
    llm = LLMClient(
        provider="openai",
        model="gpt-5-mini",
        extra_body={"safety_identifier": "user-123"},
        extra_headers={"X-Client-Team": "billing"},
    )
    print(f"  to_dict(): {llm.to_dict()}   ← no extras, no headers")
    print(f"  reply: {llm.complete([UserMessage('Say OK.')]).content}")


if __name__ == "__main__":
    show_model_info()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — skipping the live parts")
    else:
        show_fallback()
        show_passthrough()
