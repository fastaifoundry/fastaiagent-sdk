"""
Wrap a CrewAI crew with the FastAIAgent harness.

Same pattern as ``langchain_example.py``:

    1. Build your CrewAI crew normally (no FastAIAgent in sight).
    2. ``ca_int.enable()`` once — every CrewAI ``kickoff()`` is now
       auto-traced into ``.fastaiagent/local.db``.
    3. Use ``ca_int.kb_as_tool("support-kb")`` to plug the same
       FastAIAgent ``LocalKB`` into the crew as a native CrewAI
       ``BaseTool``.
    4. Use ``ca_int.prompt_from_registry("support-prompt")`` to load
       the FastAIAgent ``PromptRegistry`` slug as a raw template
       string you assign to your CrewAI ``Agent``'s ``backstory`` /
       ``system_template`` field.
    5. ``ca_int.with_guardrails(crew, ...)`` returns a wrapper whose
       ``kickoff()`` runs your input + output guardrails around the
       crew's normal ``kickoff()``.
    6. ``ca_int.register_agent(crew, name="...")`` files an entry
       in ``external_agents`` so the Local UI's /agents view lists
       the crew alongside native fa.Agents with a ``framework: crewai``
       badge.

Run:
    pip install fastaiagent crewai
    OPENAI_API_KEY=sk-... python crewai_example.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — skipping.")
        return 0
    try:
        from crewai import Agent as CrewAgent
        from crewai import Crew, Process, Task
    except ImportError:
        print("Install: pip install crewai")
        return 0

    from fastaiagent.integrations import crewai as ca_int

    from shared.guardrails import input_guardrails, output_guardrails
    from shared.kb import support_kb  # noqa: F401  — ensures the KB is materialized
    from shared.prompts import register_support_prompt

    # 1. Auto-trace every CrewAI kickoff.
    ca_int.enable()

    # 2. Register the prompt; reuse the same registered slug across
    #    all three frameworks. CrewAI consumes it as a raw string.
    system_prompt = register_support_prompt()

    # 3. KB as a CrewAI tool. The crew's agent calls it whenever the
    #    LLM decides the task needs grounded support content.
    kb_tool = ca_int.kb_as_tool(
        "support-kb",
        top_k=3,
        description="Search the support knowledge base for FAQ answers.",
        agent="ca-support-bot",
    )

    # 4. Build the crew. CrewAI doesn't have a single "system_prompt"
    #    field — the registered text goes into the agent's ``backstory``
    #    so it shapes every task this agent performs.
    support_agent = CrewAgent(
        role="Customer Support Specialist",
        goal="Answer customer questions accurately, grounded in the FAQ.",
        backstory=system_prompt,
        tools=[kb_tool],
        llm=f"openai/{os.getenv('LLM_MODEL', 'gpt-4o-mini')}",
        allow_delegation=False,
        verbose=False,
    )

    answer_task = Task(
        description="Answer this customer question: {question}",
        expected_output="A concise 2-3 sentence answer grounded in the FAQ.",
        agent=support_agent,
    )

    crew = Crew(
        agents=[support_agent],
        tasks=[answer_task],
        process=Process.sequential,
        verbose=False,
    )

    # 5. Wrap with FastAIAgent guardrails. The wrapper proxies kickoff().
    guarded = ca_int.with_guardrails(
        crew,
        name="ca-support-bot",
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
    )

    # 6. Register in the external_agents table.
    ca_int.register_agent(guarded, name="ca-support-bot")

    # 7. Run the crew. Same kickoff() contract as a bare Crew.
    print("\n— CrewAI + FastAIAgent harness —")
    result = guarded.kickoff(inputs={"question": "What is the refund window?"})
    answer = getattr(result, "raw", str(result))
    print("Q: What is the refund window?")
    print(f"A: {answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
