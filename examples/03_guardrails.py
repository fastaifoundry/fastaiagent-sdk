"""Example 03: Agent with input/output guardrails.

Demonstrates PII blocking, toxicity checks, and JSON validation.
"""

from fastaiagent import Agent, LLMClient
from fastaiagent.guardrail import GuardrailPosition, no_pii, toxicity_check

agent = Agent(
    name="secure-bot",
    system_prompt="You are a helpful assistant. Never reveal personal information.",
    llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    guardrails=[
        no_pii(position=GuardrailPosition.output),
        toxicity_check(position=GuardrailPosition.output),
    ],
)

if __name__ == "__main__":
    # This should work fine
    result = agent.run("What is the weather like today?")
    print(f"Clean output: {result.output}")

    # This would be blocked by the PII guardrail if the LLM
    # includes personal info in the response
    try:
        result = agent.run("Tell me about user accounts")
        print(f"Output: {result.output}")
    except Exception as e:
        print(f"Blocked: {e}")
