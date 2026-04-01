"""Example 08: Trace a LangChain agent with FastAIAgent.

Shows how to enable auto-tracing for LangChain agents.
Requires: pip install fastaiagent[langchain]
"""

# Enable tracing BEFORE importing LangChain
# import fastaiagent
# fastaiagent.integrations.langchain.enable()

# Then use LangChain as normal — all calls are traced
# from langchain.agents import create_tool_calling_agent
# agent = create_tool_calling_agent(...)
# result = agent.invoke({"input": "Hello"})

# Traces are stored locally in .fastaiagent/traces.db
# View them with: fastaiagent traces list

if __name__ == "__main__":
    print("LangChain auto-tracing example")
    print("=" * 40)
    print()
    print("1. Install: pip install fastaiagent[langchain]")
    print("2. Add to your code:")
    print("   import fastaiagent")
    print("   fastaiagent.integrations.langchain.enable()")
    print("3. Run your LangChain agent as normal")
    print("4. View traces: fastaiagent traces list")
