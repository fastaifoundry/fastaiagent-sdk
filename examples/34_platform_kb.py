"""Example 34: Agent with a platform-hosted Knowledge Base.

Uses ``fa.PlatformKB`` to retrieve from a KB hosted on the FastAIAgent
platform. The platform runs the full retrieval pipeline (hybrid search,
reranking, relevance gate) — the SDK is a thin client.

Required environment:
    FASTAIAGENT_API_KEY  — API key with ``kb:read`` scope
    FASTAIAGENT_TARGET   — platform URL (defaults to https://app.fastaiagent.net)
    FA_TEST_KB_ID        — UUID of a seeded KB accessible to the API key
    OPENAI_API_KEY       — for the agent's LLM

Contrast with example 06 (``LocalKB``), which keeps the KB on disk. The
wiring on ``Agent`` is identical — both expose ``.as_tool()``.
"""

from __future__ import annotations

import os

import fastaiagent as fa

fa.connect(
    api_key=os.environ["FASTAIAGENT_API_KEY"],
    target=os.environ.get("FASTAIAGENT_TARGET", "https://app.fastaiagent.net"),
)

kb = fa.PlatformKB(kb_id=os.environ["FA_TEST_KB_ID"])

agent = fa.Agent(
    name="policy-bot",
    system_prompt=(
        "You are a customer support agent. "
        "Use the search tool to find relevant information before answering."
    ),
    llm=fa.LLMClient(provider="openai", model="gpt-4o-mini"),
    tools=[kb.as_tool()],
)

if __name__ == "__main__":
    # Direct retrieval — bypasses the LLM, useful for inspection.
    print("--- Direct PlatformKB.search ---")
    for r in kb.search("refund policy", top_k=3):
        print(f"  [{r.score:.3f}] {r.chunk.content[:120]}...")

    # End-to-end: agent retrieves and answers from the KB.
    print("\n--- Agent run ---")
    result = agent.run("What's our refund policy?")
    print(result.output)
    if result.trace_id:
        print(f"\ntrace_id: {result.trace_id}")
