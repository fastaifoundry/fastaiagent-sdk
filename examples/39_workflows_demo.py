"""Example 39 — Workflows demo: Chain, Swarm, and Supervisor in one run.

Produces real traces for all three workflow runners so the Local UI's
**Workflows** page (`/workflows`) has something meaningful to show.

What it does:

  1. **Chain** — ``content-pipeline``: 3-node chain (classifier → researcher
     → writer). Runs 3 times with different topics.
  2. **Swarm** — ``support-triage``: a triage agent that hands off to either
     a coder or a writer based on the query. Runs 3 times.
  3. **Supervisor** — ``incident-commander``: a manager agent that delegates
     to a logs-analyst and a mitigation-planner worker. Runs 2 times.

Total: 8 workflow traces, plus their nested agent/LLM spans. Costs a
few cents on ``gpt-4o-mini``.

Prereqs:
    pip install 'fastaiagent[ui,openai]'
    export OPENAI_API_KEY=sk-...

Run:
    python examples/39_workflows_demo.py
    fastaiagent ui       # open http://127.0.0.1:7842/workflows
"""

from __future__ import annotations

import os
import sys

from fastaiagent import Agent, LLMClient, Supervisor, Worker
from fastaiagent.agent.swarm import Swarm
from fastaiagent.chain import Chain


def section(title: str) -> None:
    print()
    print(f"── {title} ".ljust(72, "─"))


def require_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for this example.")
        sys.exit(0)


# ─── Chain ─────────────────────────────────────────────────────────────────


def build_chain(llm: LLMClient) -> Chain:
    classifier = Agent(
        name="classifier",
        system_prompt=(
            "Classify the topic into one of: technology, science, history, business. "
            "Reply with just the category word."
        ),
        llm=llm,
    )
    researcher = Agent(
        name="researcher",
        system_prompt=(
            "Given a topic and its category, list 3 key facts. Use bullet points. "
            "Be concise."
        ),
        llm=llm,
    )
    writer = Agent(
        name="writer",
        system_prompt=(
            "Write a single tight paragraph (≤80 words) from the bullet points. "
            "No preamble."
        ),
        llm=llm,
    )

    chain = Chain(name="content-pipeline")
    chain.add_node("classify", agent=classifier)
    chain.add_node("research", agent=researcher)
    chain.add_node("write", agent=writer)
    chain.connect("classify", "research")
    chain.connect("research", "write")
    return chain


def run_chain(chain: Chain) -> None:
    section("Chain: content-pipeline")
    for topic in [
        "The history of SQLite",
        "How FAISS vector indexes work",
        "The Roman aqueducts",
    ]:
        print(f"► {topic}")
        result = chain.execute({"message": topic})
        print(f"  output: {str(result.output)[:140]}")


# ─── Swarm ─────────────────────────────────────────────────────────────────


def build_swarm(llm: LLMClient) -> Swarm:
    triage = Agent(
        name="triage",
        system_prompt=(
            "You route requests. If the user asks about code, hand off to 'coder'. "
            "If they ask about writing or prose, hand off to 'writer'. "
            "Otherwise answer briefly yourself."
        ),
        llm=llm,
    )
    coder = Agent(
        name="coder",
        system_prompt=(
            "You are a senior Python engineer. Answer code questions concisely "
            "with a short example. Do not hand off."
        ),
        llm=llm,
    )
    writer = Agent(
        name="writer",
        system_prompt=(
            "You are a prose editor. Help with writing and clarity. "
            "Answer in 2 sentences. Do not hand off."
        ),
        llm=llm,
    )
    return Swarm(
        name="support-triage",
        agents=[triage, coder, writer],
        entrypoint="triage",
        handoffs={
            "triage": ["coder", "writer"],
            "coder": [],
            "writer": [],
        },
        max_handoffs=2,
    )


def run_swarm(swarm: Swarm) -> None:
    section("Swarm: support-triage")
    for prompt in [
        "How do I reverse a list in Python without mutating it?",
        "Edit this sentence for clarity: 'The quick brown fox jumps over the lazy dog.'",
        "What's 2+2?",
    ]:
        print(f"► {prompt}")
        result = swarm.run(prompt)
        handoffs = [
            c["tool_name"] for c in result.tool_calls
            if c.get("tool_name", "").startswith("handoff_to_")
        ]
        print(f"  handoffs: {handoffs}")
        print(f"  output:   {result.output[:140]}")


# ─── Supervisor ────────────────────────────────────────────────────────────


def build_supervisor(llm: LLMClient) -> Supervisor:
    logs_agent = Agent(
        name="logs-analyst",
        system_prompt=(
            "You analyze log snippets and identify anomalies. Reply in 2 sentences."
        ),
        llm=llm,
    )
    mitigator_agent = Agent(
        name="mitigation-planner",
        system_prompt=(
            "Given an incident description, propose 2 short mitigation steps. "
            "Bullet points only."
        ),
        llm=llm,
    )
    # Role defaults to agent.name (alphanumeric-safe); description is
    # surfaced to the supervisor LLM when choosing which worker to call.
    return Supervisor(
        name="incident-commander",
        llm=llm,
        workers=[
            Worker(agent=logs_agent, description="Analyze logs for the incident."),
            Worker(agent=mitigator_agent, description="Propose mitigation steps."),
        ],
    )


def run_supervisor(supervisor: Supervisor) -> None:
    section("Supervisor: incident-commander")
    for prompt in [
        "Spike in 500 errors from the /checkout endpoint at 14:03. "
        "Walk me through analysis and mitigation.",
        "Auth latency doubled since deploy. What do we do?",
    ]:
        print(f"► {prompt[:80]}")
        result = supervisor.run(prompt)
        print(f"  output: {result.output[:140]}")


# ─── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    require_key()
    llm = LLMClient(provider="openai", model="gpt-4o-mini")

    run_chain(build_chain(llm))
    run_swarm(build_swarm(llm))
    run_supervisor(build_supervisor(llm))

    section("Done")
    print("All three workflow runners produced traces. Open the UI:")
    print("  fastaiagent ui")
    print()
    print("Browse to http://127.0.0.1:7842/workflows to see the directory,")
    print("or drill into a specific one:")
    print("  /workflows/chain/content-pipeline")
    print("  /workflows/swarm/support-triage")
    print("  /workflows/supervisor/incident-commander")


if __name__ == "__main__":
    main()
