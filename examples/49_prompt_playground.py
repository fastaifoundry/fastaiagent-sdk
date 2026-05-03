"""Example 49 — Prompt Playground demo.

Seeds the prompt registry with a couple of templates so the Local UI's
Playground page has something to select. Then prints the URL to open.

Prereqs:
    pip install 'fastaiagent[ui,openai]'
    export OPENAI_API_KEY=sk-...

Run:
    python examples/49_prompt_playground.py
    fastaiagent ui
    # Open http://127.0.0.1:7842/playground

The Playground is for the inner iteration loop: pick a prompt, fill its
variables, run the LLM call, see the streamed response inline. No
script-edit-rerun cycle. Saved cases land as JSONL under
``./.fastaiagent/datasets/`` so the eval framework can load them
unchanged.
"""

from __future__ import annotations

import sys

from fastaiagent.prompt.registry import PromptRegistry


def main() -> int:
    reg = PromptRegistry()  # uses the project's local registry

    reg.register(
        name="support-greeting",
        template=(
            "You are a friendly support agent for {{company}}.\n"
            "A customer named {{customer_name}} asks about {{topic}}.\n"
            "Reply in 2–3 sentences, polite and concrete."
        ),
        metadata={
            "purpose": "Customer support opener",
            "owner": "support-team",
        },
    )

    reg.register(
        name="code-review-summary",
        template=(
            "Summarise the following code-review comments in one paragraph "
            "for the author of the PR:\n\n{{comments}}"
        ),
        metadata={"purpose": "PR review summary"},
    )

    reg.register(
        name="image-describe",
        template=(
            "Describe what you see in the attached image. "
            "Mention any text visible in the image. "
            "Pay particular attention to {{focus}}."
        ),
        metadata={"purpose": "Vision-model image describer"},
    )

    print("Registered 3 prompts: support-greeting, code-review-summary, image-describe")
    print()
    print("Next steps:")
    print("  1. Start the UI:        fastaiagent ui --no-auth")
    print("  2. Open the Playground: http://127.0.0.1:7842/playground")
    print("  3. Pick 'support-greeting', fill {{company}}/{{customer_name}}/{{topic}},")
    print("     choose gpt-4o-mini, click Run. Tokens stream in.")
    print("  4. Click 'Save as eval case' to bridge the run into the eval framework:")
    print("     the file lands at ./.fastaiagent/datasets/<name>.jsonl and is")
    print("     loadable via Dataset.from_jsonl().")
    print()
    print("Tip: from /prompts/<slug>, the 'Test in Playground' button pre-loads")
    print("the prompt and version into the Playground for you.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
