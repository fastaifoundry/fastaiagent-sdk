# Provider Support

The SDK covers five providers for multimodal input. Each provider has its
own wire format; `LLMClient` abstracts that away — the developer just
passes `Image` / `PDF` and the SDK rewrites for the active provider.

## Capability matrix

| Provider          | Image | PDF text | PDF vision (rendered) | PDF native | Tool returns image |
|-------------------|:-----:|:--------:|:---------------------:|:----------:|:------------------:|
| OpenAI gpt-4o     | Yes   | Yes      | Yes                   | —          | Yes (auto-split)   |
| Azure gpt-4o      | Yes   | Yes      | Yes                   | —          | Yes (auto-split)   |
| Anthropic 3.5+    | Yes   | Yes      | Yes                   | Yes        | Yes (block)        |
| Bedrock (Claude)  | Yes   | Yes      | Yes                   | Yes        | Yes (block)        |
| Ollama (vision)   | Yes   | Yes      | Yes                   | —          | Yes (auto-split)   |
| Vertex / Gemini   | (deferred) | (deferred) | (deferred)        | —          | (deferred)         |

Vertex/Gemini support is tracked separately — `LLMClient` does not yet
implement that provider. Open an issue if you need it.

## Vision-capable model registry

The SDK keeps a small prefix-based registry of vision-capable models:

```python
from fastaiagent.multimodal import is_vision_capable, supports_native_pdf

is_vision_capable("openai", "gpt-4o-mini")              # True
is_vision_capable("openai", "gpt-3.5-turbo")             # False
is_vision_capable("anthropic", "claude-sonnet-4-6")      # True
supports_native_pdf("anthropic", "claude-sonnet-4-6")    # True
```

Sending an `Image` (or a PDF in vision mode) to a non-vision model
raises `NonVisionModelError` *before* any HTTP request — the SDK fails
early with a clear message rather than letting the provider 4xx.

## Auto-detection rules (`pdf_mode="auto"`)

1. If the model supports **native** PDF (Anthropic Sonnet/Opus 3.5+,
   Bedrock-hosted Claude 3.5+) → emit a `document` block.
2. Else if the model is **vision-capable** → render pages with pymupdf
   and emit per-page image blocks.
3. Else → extract text with pymupdf and emit a single text block.

You can always override with `pdf_mode="text" | "vision" | "native"` on
the `LLMClient` or via `fa.config.pdf_mode`.

## OpenAI quirk: tool messages can't carry images

OpenAI rejects `image_url` blocks inside `role="tool"` messages. When a
tool returns `Image` or `PDF` and the active provider is OpenAI/Azure/
Custom, the SDK transparently splits the tool result into:

* `tool` message with the textual summary
* a follow-up `user` message carrying the multimodal payload

Anthropic, Bedrock, and Ollama accept images directly inside tool-result
blocks, so no split happens there.
