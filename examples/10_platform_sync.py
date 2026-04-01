"""Example 10: Push agents/chains to FastAIAgent Platform.

Shows how to sync SDK resources to the platform for
visual editing, monitoring, and team collaboration.

NOTE: Push-only sync (SDK → Platform). Phase 40 feature.
"""

# from fastaiagent import Agent, Chain, FastAI, LLMClient
#
# # Connect to platform
# fa = FastAI(api_key="sk-...", project="customer-support")
#
# # Define agent locally
# agent = Agent(
#     name="support-bot",
#     system_prompt="You are a helpful support agent.",
#     llm=LLMClient(provider="openai", model="gpt-4o"),
# )
#
# # Push to platform — appears in the visual editor
# result = fa.push(agent)
# print(f"Pushed: {result.url}")
#
# # Traces automatically sent to platform dashboard
# with fa.trace("support-session"):
#     output = agent.run("How do I get a refund?")

if __name__ == "__main__":
    print("Platform Sync example (Phase 40)")
    print("=" * 40)
    print()
    print("SDK → Platform push will be available in Phase 40.")
    print("The SDK works fully standalone without platform connection.")
    print()
    print("When available:")
    print("  fa = FastAI(api_key='sk-...', project='my-project')")
    print("  fa.push(my_agent)    # Push agent to platform")
    print("  fa.push(my_chain)    # Push chain to visual editor")
