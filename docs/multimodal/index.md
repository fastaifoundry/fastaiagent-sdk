# Multimodal Input

FastAIAgent agents accept images, PDFs, and **any file/bytes** as first-class
inputs alongside text. The same code works against OpenAI, Azure, Anthropic,
Gemini, Bedrock, and Ollama — the SDK handles the provider-specific wire
formatting and forwards files **natively** (never stringified, never rendered
locally when the provider can parse them).

```python
from fastaiagent import Agent, LLMClient, Image, PDF, File

agent = Agent(
    name="claims",
    llm=LLMClient(provider="anthropic", model="claude-sonnet-4-6"),
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

Or just hand the agent bytes / a path — the type is sniffed and wrapped in a
[`File`](files.md) automatically:

```python
agent.run(file_bytes)                       # mime sniffed from the bytes
agent.run(["Summarize this.", Path("q3.csv")])
```

## What's supported

| Capability                                | Status |
|-------------------------------------------|--------|
| Image input (JPEG, PNG, GIF, WebP)        | Yes    |
| PDF input (text / vision / native modes)  | Yes    |
| Native PDF (OpenAI/Azure `file`, Anthropic/Bedrock `document`, Gemini `inlineData`) | Yes |
| Generic `File` input (any bytes, mime-routed) | Yes |
| Auto-detect bare `bytes` / `Path`         | Yes    |
| Documents (csv/txt/html/md, docx/xlsx via Bedrock/Gemini) | Yes |
| Audio (wav/mp3 on OpenAI/Gemini)          | Yes    |
| Tool returns `Image` / `PDF`              | Yes    |
| Multimodal in Chain state (checkpointed)  | Yes    |
| Multimodal in Swarm shared context        | Yes    |
| Multimodal in eval JSONL datasets         | Yes    |
| Multimodal in trace inspector + Replay    | Yes    |

Input-only: image/audio generation and OCR-as-a-feature are out of scope.
Which file types a given provider accepts natively varies — see the
[File routing matrix](files.md#what-each-provider-accepts-natively).

## Read next

* **[When to use it](when-to-use.md)** — read this first if you're
  weighing the abstraction against just base64-encoding the file
  yourself. Honest decision rubric for when the layer earns its keep
  and when it's overkill.
* [Images](images.md) — `Image` constructors, supported formats, sizes, the OpenAI `detail` parameter.
* [PDFs](pdfs.md) — `PDF` constructors, text vs vision vs native modes.
* [Files (any type)](files.md) — the generic `File` input, auto-detect, and the per-provider routing matrix.
* [Provider Support](providers.md) — capability matrix and auto-detection rules.
* [Multimodal Eval](eval.md) — JSONL syntax for image/PDF references.
