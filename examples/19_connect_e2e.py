"""Example 19: Full fa.connect() end-to-end demo.

Publishes prompts to the platform, creates agents with OpenAI and Anthropic,
fetches prompts from platform, executes agents, and verifies traces landed.

Usage:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export FASTAIAGENT_API_KEY=fa_k_...
    export FASTAIAGENT_TARGET=http://localhost:8001
    python examples/19_connect_e2e.py
"""

import os
import time

import httpx

import fastaiagent as fa
from fastaiagent import Agent, FunctionTool, LLMClient
from fastaiagent.client import _connection
from fastaiagent.eval import Dataset, EvalResults, evaluate
from fastaiagent.eval.scorer import Scorer, ScorerResult
from fastaiagent.prompt import PromptRegistry
from fastaiagent.trace.replay import Replay


# ── Tools ──────────────────────────────────────────────────────────────────────


def lookup_order(order_id: str) -> str:
    """Look up an order by ID."""
    orders = {
        "ORD-001": "MacBook Pro 16-inch, shipped 2026-04-01, delivered 2026-04-03",
        "ORD-002": "AirPods Pro, processing, estimated delivery 2026-04-10",
        "ORD-003": "iPad Air, cancelled by customer on 2026-03-28",
    }
    return orders.get(order_id, f"Order {order_id} not found.")


def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.environ.get("FASTAIAGENT_API_KEY", "")
    target = os.environ.get("FASTAIAGENT_TARGET", "http://localhost:8001")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        print("Skipping: FASTAIAGENT_API_KEY not set")
        raise SystemExit(1)
    if not openai_key:
        print("Skipping: OPENAI_API_KEY not set")
        raise SystemExit(1)

    # ════════════════════════════════════════════
    #  Step 1: Connect to platform
    # ════════════════════════════════════════════

    print("=" * 60)
    print("  Step 1: Connect to FastAIAgent Platform")
    print("=" * 60)

    fa.connect(api_key=api_key, target=target)
    print(f"  Connected: {fa.is_connected}")
    print(f"  Domain:    {_connection.domain_id}")
    print(f"  Project:   {_connection.project_id}")
    print(f"  Scopes:    {len(_connection.scopes)} scopes")
    print()

    # ════════════════════════════════════════════
    #  Step 2: Publish prompts to platform
    # ════════════════════════════════════════════

    print("=" * 60)
    print("  Step 2: Publish prompts to platform")
    print("=" * 60)

    registry = PromptRegistry()

    registry.publish(
        slug="support-agent-prompt",
        content=(
            "You are a customer support agent for {{company}}.\n"
            "Be helpful, concise, and professional.\n"
            "Use tools to look up order information when asked."
        ),
        variables=["company"],
    )
    print("  Published: support-agent-prompt")

    registry.publish(
        slug="math-tutor-prompt",
        content=(
            "You are a math tutor for {{school}}.\n"
            "Explain step by step. Use the calculate tool for verification.\n"
            "Keep answers under 3 sentences."
        ),
        variables=["school"],
    )
    print("  Published: math-tutor-prompt")
    print()

    # ════════════════════════════════════════════
    #  Step 3: Fetch prompts from platform
    # ════════════════════════════════════════════

    print("=" * 60)
    print("  Step 3: Fetch prompts from platform")
    print("=" * 60)

    support_prompt = registry.get("support-agent-prompt", source="platform")
    print(f"  Fetched: {support_prompt.name} v{support_prompt.version}")
    print(f"  Variables: {support_prompt.variables}")

    math_prompt = registry.get("math-tutor-prompt", source="platform")
    print(f"  Fetched: {math_prompt.name} v{math_prompt.version}")
    print()

    # ════════════════════════════════════════════
    #  Step 4: Create LLM clients
    # ════════════════════════════════════════════

    print("=" * 60)
    print("  Step 4: Create LLM clients")
    print("=" * 60)

    openai_llm = LLMClient(provider="openai", model="gpt-4.1")
    print("  Created: OpenAI gpt-4.1")

    if anthropic_key:
        anthropic_llm = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
        print("  Created: Anthropic claude-sonnet-4-20250514")
    else:
        anthropic_llm = None
        print("  Skipped: Anthropic (ANTHROPIC_API_KEY not set)")
    print()

    # ════════════════════════════════════════════
    #  Step 5: Create agents with platform prompts
    # ════════════════════════════════════════════

    print("=" * 60)
    print("  Step 5: Create agents with platform prompts")
    print("=" * 60)

    # Resolve prompt variables
    support_system = support_prompt.format(company="Acme Corp")
    math_system = math_prompt.format(school="Springfield Academy")

    support_agent = Agent(
        name="support-bot-openai",
        system_prompt=support_system,
        llm=openai_llm,
        tools=[FunctionTool(name="lookup_order", fn=lookup_order)],
    )
    print(f"  Created: {support_agent.name} (OpenAI + lookup_order tool)")

    math_agent = Agent(
        name="math-tutor-openai",
        system_prompt=math_system,
        llm=openai_llm,
        tools=[FunctionTool(name="calculate", fn=calculate)],
    )
    print(f"  Created: {math_agent.name} (OpenAI + calculate tool)")

    if anthropic_llm:
        support_agent_claude = Agent(
            name="support-bot-claude",
            system_prompt=support_system,
            llm=anthropic_llm,
            tools=[FunctionTool(name="lookup_order", fn=lookup_order)],
        )
        print(f"  Created: {support_agent_claude.name} (Anthropic + lookup_order tool)")
    print()

    # ════════════════════════════════════════════
    #  Step 6: Execute agents (traces auto-sent)
    # ════════════════════════════════════════════

    print("=" * 60)
    print("  Step 6: Execute agents — traces auto-sent to platform")
    print("=" * 60)

    trace_ids = []

    # Run support agent (OpenAI)
    print("\n  [OpenAI] Support: 'What's the status of order ORD-001?'")
    r1 = support_agent.run("What's the status of order ORD-001?")
    print(f"  Output: {r1.output}")
    print(f"  Trace:  {r1.trace_id} | Tokens: {r1.tokens_used} | Latency: {r1.latency_ms}ms")
    trace_ids.append(r1.trace_id)

    # Run math agent (OpenAI)
    print("\n  [OpenAI] Math: 'What is 17 * 23 + 45?'")
    r2 = math_agent.run("What is 17 * 23 + 45?")
    print(f"  Output: {r2.output}")
    print(f"  Trace:  {r2.trace_id} | Tokens: {r2.tokens_used} | Latency: {r2.latency_ms}ms")
    trace_ids.append(r2.trace_id)

    # Run support agent (Anthropic)
    if anthropic_llm:
        print("\n  [Anthropic] Support: 'Check order ORD-002 and ORD-003'")
        r3 = support_agent_claude.run("Check order ORD-002 and ORD-003 for me.")
        print(f"  Output: {r3.output}")
        print(f"  Trace:  {r3.trace_id} | Tokens: {r3.tokens_used} | Latency: {r3.latency_ms}ms")
        trace_ids.append(r3.trace_id)

    print()

    # ════════════════════════════════════════════
    #  Step 7: Publish eval dataset + run eval + publish results
    # ════════════════════════════════════════════

    print("=" * 60)
    print("  Step 7: Evaluate & publish results")
    print("=" * 60)

    # Create dataset
    dataset = Dataset.from_list([
        {"input": "What's the status of ORD-001?", "expected": "shipped"},
        {"input": "What's the status of ORD-002?", "expected": "processing"},
        {"input": "What's the status of ORD-003?", "expected": "cancelled"},
        {"input": "What's the status of ORD-999?", "expected": "not found"},
    ])

    # Publish dataset to platform
    dataset.publish("support-bot-golden-set")
    print("  Published dataset: support-bot-golden-set (4 items)")

    # Run eval locally
    def support_fn(input_text: str) -> str:
        return support_agent.run(input_text, trace=False).output

    @Scorer.code("contains_keyword")
    def contains_keyword(input: str, output: str, expected: str | None = None) -> ScorerResult:
        if expected and expected.lower() in output.lower():
            return ScorerResult(score=1.0, passed=True, reason=f"Contains '{expected}'")
        return ScorerResult(score=0.0, passed=False, reason=f"Missing '{expected}'")

    results = evaluate(agent_fn=support_fn, dataset=dataset, scorers=[contains_keyword])
    print(f"\n  {results.summary()}")

    # Publish results to platform
    results.publish(run_name="support-bot-v1-golden")
    print("\n  Published eval run: support-bot-v1-golden")
    print()

    # ════════════════════════════════════════════
    #  Step 8: Flush traces & verify on platform
    # ════════════════════════════════════════════

    print("=" * 60)
    print("  Step 8: Flush traces & verify on platform")
    print("=" * 60)

    # Force flush all pending spans
    fa.disconnect()
    print("  Disconnected (spans flushed)")

    # Brief pause for platform to process
    time.sleep(1)

    # Verify traces via platform API
    print(f"\n  Verifying {len(trace_ids)} traces on platform...")
    for tid in trace_ids:
        if not tid:
            print(f"    {tid}: SKIPPED (no trace ID)")
            continue
        resp = httpx.get(
            f"{target}/public/v1/traces/{tid}",
            headers={"X-API-Key": api_key},
        )
        if resp.status_code == 200:
            data = resp.json()
            span_count = len(data.get("spans", []))
            spans = ", ".join(s["name"] for s in data.get("spans", []))
            print(f"    {tid}: ✓ status={data['status']} spans={span_count} [{spans}]")
        else:
            print(f"    {tid}: ✗ {resp.status_code} {resp.text[:80]}")

    print()
    print("=" * 60)
    print("  ALL DONE — check platform dashboard for traces, prompts, eval")
    print("=" * 60)
