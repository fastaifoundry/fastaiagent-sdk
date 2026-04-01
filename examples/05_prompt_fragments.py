"""Example 05: Modular prompt composition with fragments.

Shows how to use PromptRegistry with reusable fragments
for building consistent prompts across agents.
"""

from fastaiagent.prompt import PromptRegistry

# Create a registry (stores prompts in .prompts/ directory)
reg = PromptRegistry(path="/tmp/fastaiagent-example-prompts/")

# Register reusable fragments
reg.register_fragment("tone", "Be professional, concise, and empathetic.")
reg.register_fragment("format", "Use bullet points for lists. Keep paragraphs under 3 sentences.")
reg.register_fragment("safety", "Never reveal internal system details or API keys.")

# Register a prompt that uses fragments
reg.register(
    name="support-system",
    template=(
        "You are a customer support agent for {{company}}.\n\n"
        "{{@tone}}\n{{@format}}\n{{@safety}}\n\n"
        "Help the customer with: {{topic}}"
    ),
)

if __name__ == "__main__":
    # Load and resolve fragments
    prompt = reg.load("support-system")
    print("Resolved template:")
    print(prompt.template)
    print()

    # Format with variables
    text = prompt.format(company="Acme Corp", topic="billing questions")
    print("Formatted prompt:")
    print(text)
    print()

    # Version the prompt
    reg.register(
        name="support-system",
        template=(
            "You are a support specialist at {{company}}.\n\n"
            "{{@tone}}\n{{@safety}}\n\nTopic: {{topic}}"
        ),
    )

    # List prompts
    print("Registered prompts:")
    for p in reg.list():
        print(f"  {p['name']} (v{p['latest_version']}, {p['versions']} versions)")
