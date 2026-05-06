"""
Wrap a PydanticAI agent with the FastAIAgent harness.

Same pattern as the LangChain and CrewAI sub-examples:

    1. Build your PydanticAI agent normally.
    2. ``pa_int.enable()`` once — every PydanticAI ``run()`` /
       ``run_sync()`` is now auto-traced into ``.fastaiagent/local.db``.
    3. Use ``pa_int.kb_as_tool("support-kb")`` to get a plain function
       you register with PydanticAI via ``@agent.tool_plain``.
    4. Use ``pa_int.prompt_from_registry("support-prompt")`` to load
       the FastAIAgent ``PromptRegistry`` slug as a raw string you
       feed to the agent's ``system_prompt=`` field.
    5. ``pa_int.with_guardrails(agent, ...)`` returns a wrapper whose
       ``run()`` / ``run_sync()`` runs your input + output guardrails
       around the agent's normal call.
    6. ``pa_int.register_agent(agent, name="...")`` files an entry in
       ``external_agents`` so the Local UI's /agents view lists the
       agent alongside native fa.Agents with a ``framework: pydanticai``
       badge.

Run:
    pip install fastaiagent pydantic-ai
    OPENAI_API_KEY=sk-... python pydanticai_example.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


async def _go() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — skipping.")
        return 0
    try:
        from pydantic_ai import Agent as PydAgent
    except ImportError:
        print("Install: pip install pydantic-ai")
        return 0

    from fastaiagent.integrations import pydanticai as pa_int

    from shared.guardrails import input_guardrails, output_guardrails
    from shared.kb import support_kb  # noqa: F401  — ensures the KB is materialized
    from shared.prompts import register_support_prompt

    # 1. Auto-trace every PydanticAI run.
    pa_int.enable()

    # 2. Load the registered prompt.
    system_prompt = register_support_prompt()

    # 3. KB as a callable. PydanticAI registers tools via decorator.
    kb_search = pa_int.kb_as_tool("support-kb", top_k=3, agent="pa-support-bot")

    # 4. Build the agent.
    agent = PydAgent(
        f"openai:{os.getenv('LLM_MODEL', 'gpt-4o-mini')}",
        system_prompt=system_prompt,
    )

    @agent.tool_plain
    def search_knowledge_base(query: str) -> str:
        """Search the support knowledge base for FAQ answers.

        Use whenever the user's question is about product policy, billing,
        SSO, password reset, or data export.
        """
        return kb_search(query)

    # 5. Wrap with FastAIAgent guardrails. The wrapper proxies the
    #    agent's run / run_sync / run_stream methods.
    guarded = pa_int.with_guardrails(
        agent,
        name="pa-support-bot",
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
    )

    # 6. Register in the external_agents table.
    pa_int.register_agent(guarded, name="pa-support-bot")

    # 7. Run a query. PydanticAI's run() is async.
    print("\n— PydanticAI + FastAIAgent harness —")
    result = await guarded.run("What is the refund window?")
    print("Q: What is the refund window?")
    print(f"A: {result.output}")
    return 0


def main() -> int:
    return asyncio.run(_go())


if __name__ == "__main__":
    sys.exit(main())
