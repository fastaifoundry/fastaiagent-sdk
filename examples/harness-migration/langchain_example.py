"""
Wrap a LangGraph agent with the FastAIAgent harness.

This shows the canonical migration pattern:

    1. Build your LangGraph agent normally (no FastAIAgent in sight).
    2. ``lc_int.enable()`` once — every LangChain / LangGraph chain is
       now auto-traced into ``.fastaiagent/local.db``.
    3. Use ``lc_int.kb_as_retriever("support-kb")`` to plug the same
       FastAIAgent ``LocalKB`` your fa.Agents use into the graph as a
       native LangChain ``BaseRetriever``.
    4. Use ``lc_int.prompt_from_registry("support-prompt")`` to load
       the FastAIAgent ``PromptRegistry`` slug as a native
       ``ChatPromptTemplate`` — edits in the Local UI Playground
       propagate to the LangGraph agent on its next invocation.
   5. ``lc_int.with_guardrails(graph, input_guardrails=, output_guardrails=)``
       wraps the compiled graph; it raises ``GuardrailBlocked`` when
       any check fails. Streaming through the wrapper buffers until
       output guardrails finish.
    6. ``lc_int.register_agent(graph, name="...")`` files an entry in
       ``external_agents`` so the Local UI's /agents view lists the
       LangGraph agent alongside native fa.Agents with a "framework:
       langchain" badge.

Run:
    pip install "fastaiagent[langchain]" langchain langchain-openai langgraph
    OPENAI_API_KEY=sk-... python langchain_example.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make ``shared/`` importable regardless of cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — skipping.")
        return 0
    try:
        from langchain.agents import create_agent
        from langchain_core.tools import Tool
        from langchain_openai import ChatOpenAI
    except ImportError:
        print('Install: pip install langchain langchain-openai langgraph')
        return 0

    from fastaiagent.integrations import langchain as lc_int

    from shared.guardrails import input_guardrails, output_guardrails
    from shared.kb import support_kb  # noqa: F401  — ensures the KB is materialized
    from shared.prompts import register_support_prompt

    # 1. Auto-trace every LangChain / LangGraph call.
    lc_int.enable()

    # 2. Register the prompt once. Idempotent — picks up the latest
    #    version on every run, including any UI Playground edits.
    system_prompt = register_support_prompt()

    # 3. Build the LangGraph agent. Use the FastAIAgent KB as a native
    #    LangChain retriever wrapped in a Tool the agent can call.
    retriever = lc_int.kb_as_retriever("support-kb", top_k=3, agent="lc-support-bot")

    def _search_kb(q: str) -> str:
        docs = retriever.invoke(q)
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    kb_tool = Tool(
        name="search_knowledge_base",
        description=(
            "Search the support knowledge base for relevant FAQ entries. "
            "Use whenever the user's question is about product policy, "
            "billing, SSO, password reset, or data export."
        ),
        func=_search_kb,
    )

    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
    graph = create_agent(llm, tools=[kb_tool], system_prompt=system_prompt)

    # 4. Wrap with FastAIAgent guardrails. Returned object proxies
    #    .invoke / .ainvoke / .stream / .astream — the graph's own
    #    interface is preserved.
    guarded = lc_int.with_guardrails(
        graph,
        name="lc-support-bot",
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
    )

    # 5. Register in the external_agents table so /agents shows it.
    lc_int.register_agent(guarded, name="lc-support-bot")

    # 6. Run a query. Identical to the un-wrapped LangGraph contract.
    from langchain_core.messages import HumanMessage

    print("\n— LangGraph + FastAIAgent harness —")
    result = guarded.invoke(
        {"messages": [HumanMessage(content="What is the refund window?")]}
    )
    final = result["messages"][-1]
    print("Q: What is the refund window?")
    print(f"A: {getattr(final, 'content', final)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
