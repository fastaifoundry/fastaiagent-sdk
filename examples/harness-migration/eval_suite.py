"""
Cross-framework evaluation — same dataset, three different agents.

Run: python eval_suite.py
     python eval_suite.py --framework langchain   # only one
     python eval_suite.py --publish               # also publish to platform

Demonstrates that ``fa.evaluate()`` works against any of the three
adapters via each integration's ``as_evaluable()``. The dataset, the
scorers, and the per-case reporting are identical — only the agent
factory differs. Useful for A/B-ing a prompt change across frameworks
or graduating an existing LangChain agent to native fa.Agent.

Each ``as_evaluable`` adapter:
  * langchain → returns a sync callable; ``fa.evaluate`` calls it
    directly.
  * crewai    → returns a sync callable.
  * pydanticai → returns an async callable; ``fa.evaluate`` awaits it.

All three produce the same ``_EvaluableResult`` shape (``.output: str``,
``.trace_id: str``) so the per-case ``trace_id`` correlation in
``EvalCaseRecord`` works uniformly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# Same canonical dataset every framework gets graded on. Substring matches
# (case-insensitive) — keeps the scorer cheap and deterministic.
EVAL_DATASET = [
    {"input": "What is the refund window?", "expected": "30 days"},
    {"input": "How do I reset my password?", "expected": "Settings > Security"},
    {"input": "Do you support SSO?", "expected": "Enterprise"},
    {"input": "How do I export my data?", "expected": "Settings > Data > Export"},
]


# ─── Agent factories — one per framework ────────────────────────────────────
#
# Each factory returns an `_EvaluableResult`-producing callable that
# `fa.evaluate()` can drive. They lazy-import their framework so a missing
# dep doesn't break the others.


def _make_langchain_evaluable():
    from langchain.agents import create_agent
    from langchain_core.tools import Tool
    from langchain_openai import ChatOpenAI

    from fastaiagent.integrations import langchain as lc_int

    from shared.kb import support_kb  # noqa: F401
    from shared.prompts import register_support_prompt

    lc_int.enable()
    sp = register_support_prompt()
    ret = lc_int.kb_as_retriever("support-kb", top_k=3)
    kb_tool = Tool(
        name="search_knowledge_base",
        description="Search the support FAQ.",
        func=lambda q: "\n\n".join(d.page_content for d in ret.invoke(q)),
    )
    graph = create_agent(
        ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0),
        tools=[kb_tool],
        system_prompt=sp,
    )
    return lc_int.as_evaluable(graph)


def _make_crewai_evaluable():
    from crewai import Agent as CrewAgent
    from crewai import Crew, Process, Task

    from fastaiagent.integrations import crewai as ca_int

    from shared.kb import support_kb  # noqa: F401
    from shared.prompts import register_support_prompt

    ca_int.enable()
    sp = register_support_prompt()
    kb_tool = ca_int.kb_as_tool(
        "support-kb",
        top_k=3,
        description="Search the support knowledge base for FAQ answers.",
    )
    agent = CrewAgent(
        role="Customer Support Specialist",
        goal="Answer accurately, grounded in the FAQ.",
        backstory=sp,
        tools=[kb_tool],
        llm=f"openai/{os.getenv('LLM_MODEL', 'gpt-4o-mini')}",
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description="Answer this customer question: {input}",
        expected_output="A concise grounded answer.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    # CrewAI's input mapper threads {"input": text} into kickoff(inputs=).
    return ca_int.as_evaluable(crew, input_mapper=lambda s: {"input": s})


def _make_pydanticai_evaluable():
    from pydantic_ai import Agent as PydAgent

    from fastaiagent.integrations import pydanticai as pa_int

    from shared.kb import support_kb  # noqa: F401
    from shared.prompts import register_support_prompt

    pa_int.enable()
    sp = register_support_prompt()
    kb_search = pa_int.kb_as_tool("support-kb", top_k=3)

    agent = PydAgent(
        f"openai:{os.getenv('LLM_MODEL', 'gpt-4o-mini')}",
        system_prompt=sp,
    )

    @agent.tool_plain
    def search_knowledge_base(query: str) -> str:
        """Search the support FAQ. Use for policy / billing / SSO / data export."""
        return kb_search(query)

    return pa_int.as_evaluable(agent)


_FACTORIES = {
    "langchain": _make_langchain_evaluable,
    "crewai": _make_crewai_evaluable,
    "pydanticai": _make_pydanticai_evaluable,
}


# ─── Runner ─────────────────────────────────────────────────────────────────


def run_eval(framework: str | None = None, publish: bool = False) -> None:
    import fastaiagent as fa

    frameworks = [framework] if framework else list(_FACTORIES.keys())

    for fw in frameworks:
        print(f"\n=== {fw} ===")
        try:
            evaluable = _FACTORIES[fw]()
        except ImportError as e:
            print(f"  skipped — {e}")
            continue

        results = fa.evaluate(
            evaluable,
            dataset=EVAL_DATASET,
            # ``contains`` is built-in: case-insensitive substring of expected
            # in actual_output. Cheap, deterministic, no LLM-as-judge cost.
            scorers=["contains"],
            persist=True,
            run_name=f"harness-migration-{fw}",
            dataset_name="harness-migration-faq",
            agent_name=f"{fw}-support-bot",
        )
        print(results.summary())
        if publish:
            try:
                results.publish(run_name=f"harness-migration-{fw}")
                print(f"  ✓ published {fw} run to platform")
            except Exception as e:
                print(f"  publish failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework",
        choices=list(_FACTORIES.keys()),
        help="Run only one framework. Default: all three.",
    )
    parser.add_argument("--publish", action="store_true", help="Publish to platform")
    args = parser.parse_args()
    run_eval(framework=args.framework, publish=args.publish)


if __name__ == "__main__":
    main()
