# Multimodal Input

FastAIAgent agents accept images and PDFs as first-class inputs alongside text.
The same code works against OpenAI, Azure, Anthropic, Bedrock, and Ollama —
the SDK handles the provider-specific wire formatting.

```python
from fastaiagent import Agent, LLMClient, Image, PDF

agent = Agent(
    name="claims",
    llm=LLMClient(provider="anthropic", model="claude-sonnet-4-20250514"),
)

result = agent.run(
    [
        "Compare the photo to the policy and assess the claim.",
        Image.from_file("damage.jpg"),
        PDF.from_file("policy.pdf"),
    ]
)
print(result.output)
```

## What's supported

| Capability                                | Status |
|-------------------------------------------|--------|
| Image input (JPEG, PNG, GIF, WebP)        | Yes    |
| PDF input (text mode)                     | Yes    |
| PDF input (page-as-image vision mode)     | Yes    |
| PDF input (Anthropic native `document` block) | Yes |
| Tool returns `Image` / `PDF`              | Yes    |
| Multimodal in Chain state (checkpointed)  | Yes    |
| Multimodal in Swarm shared context        | Yes    |
| Multimodal in eval JSONL datasets         | Yes    |
| Multimodal in trace inspector + Replay    | Yes    |

Audio, video, image generation, and OCR-as-a-feature are not in scope —
the spec is input-only.

## Read next

* **[When to use it](when-to-use.md)** — read this first if you're
  weighing the abstraction against just base64-encoding the file
  yourself. Honest decision rubric for when the layer earns its keep
  and when it's overkill.
* [Images](images.md) — `Image` constructors, supported formats, sizes, the OpenAI `detail` parameter.
* [PDFs](pdfs.md) — `PDF` constructors, text vs vision vs native modes.
* [Provider Support](providers.md) — capability matrix and auto-detection rules.
* [Multimodal Eval](eval.md) — JSONL syntax for image/PDF references.
