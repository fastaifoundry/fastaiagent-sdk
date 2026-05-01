# When to use the multimodal layer

`fastaiagent.Image` and `fastaiagent.PDF` are an opinionated abstraction
over what you'd otherwise build by hand: read the file, base64-encode it,
and stuff it into the right provider-specific message shape. This page
exists so you can decide whether you actually need the abstraction or
whether a six-line base64 wrapper is the right call.

## TL;DR

The bet behind the layer: providers will keep churning, and you will
eventually persist multimodal content somewhere (checkpoints, traces,
replay forks, eval datasets). It pays off when you have **≥2** of:

- Multiple LLM providers (or one today, but might switch)
- Durable workflows (Chain / Swarm checkpoints, HITL pause/resume)
- A trace store / Local UI / Agent Replay
- Eval pipelines with image or PDF test cases
- PDFs of any kind (cross-provider PDF handling is genuinely messy)

If you have none of those, you're not wrong to skip it. A one-shot
script against one provider doesn't need this layer.

## What the naive approach actually looks like

The "just base64 it" version works for one provider but is not portable:

```python
import base64

b64 = base64.b64encode(open("photo.jpg", "rb").read()).decode()

# OpenAI / Azure / Custom:
messages = [{"role": "user", "content": [
    {"type": "text", "text": "describe"},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
]}]
```

The same image becomes something different for every provider:

| Provider  | Wire format                                                                              |
|-----------|------------------------------------------------------------------------------------------|
| OpenAI    | `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}`                  |
| Azure     | same as OpenAI                                                                           |
| Anthropic | `{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"..."}}`     |
| Bedrock   | `{"image":{"format":"jpeg","source":{"bytes": <raw bytes, NOT base64>}}}`                |
| Ollama    | `{"role":"user","content":"text","images":[<base64>]}` (images at message top level) |

Switching providers means rewriting the message-build code in every place
you call the LLM. That's reason #1 the layer exists.

## Quirks the layer absorbs

A handful of provider-specific landmines that any naive integration
trips over sooner or later:

1. **OpenAI rejects `image_url` inside tool-result messages.** Tool returns
   an `Image`? Naive caller gets a 400 (`Image URLs are only allowed for
   messages with role 'user'`). The SDK silently splits the tool result
   into a (tool: text-summary, user: multimodal) pair so the LLM still
   sees the image.
2. **`gpt-3.5-turbo` refuses array-content messages outright** — even
   text-only arrays. The SDK collapses text-only blocks back to a plain
   string so non-vision OpenAI models keep accepting requests unchanged.
3. **PDFs are a three-way problem.** OpenAI / Bedrock / Ollama can't
   accept PDFs natively. You pick between `pymupdf.extract_text()`
   (cheap, layout lost), render pages to images (vision-mode, expensive,
   layout preserved), or Anthropic's native `document` block (Claude
   reads the PDF directly). `pdf_mode="auto"` picks the right one based
   on the model.
4. **Per-image size caps** vary by provider (OpenAI 20 MB, Anthropic
   5 MB). Hand a 25 MB JPEG to a naive call → 400. The SDK Pillow-resizes
   in place before sending and logs a warning.
5. **Non-vision model error before HTTP.** Sending an image to
   `gpt-3.5-turbo` naively gets you `Invalid content type. image_url is
   only supported by certain models.` — post-network, post-tokens-charged,
   confusing. The SDK raises `NonVisionModelError` synchronously, with
   the model name, before any HTTP call.

Per-call you can write fixes for these yourself. Across an agent
codebase they tend to pile into copy-paste boilerplate that drifts
between developers.

## Where the layer earns the most

Persistence is where the calculus flips from "nice" to "necessary."

| Surface          | Without the layer                                          | With the layer                                          |
|------------------|------------------------------------------------------------|---------------------------------------------------------|
| Chain checkpoint | `json.dumps(state)` blows up on `bytes`                    | `Image.to_dict()` / `from_dict()` round-trip cleanly    |
| Swarm shared state | Same — bytes don't JSON-serialize                        | Walks state through `_serialize_for_checkpoint`          |
| Trace store + UI | Trace store has to learn what an image is                  | `trace_attachments` table + thumbnail endpoint do it once |
| Replay fork      | Need a custom base64 ferry through the modify endpoint     | `forked.modify_input([..., Image.from_file("...")])` reads exactly like `agent.run(...)` |
| Eval dataset     | Hand-roll base64 in JSONL, painful diff/review             | `{"type":"image","path":"cat.jpg"}` resolved at load time |

If you have any of those surfaces, the abstraction is the cheap path. If
you don't, it's overhead.

## When NOT to use it

Be honest with yourself about scope before reaching for the layer.

- **One-shot script, one provider, no tracing.** Base64 it inline. The
  `Image` class is overhead you don't need.
- **Custom provider you wire to directly.** You may already speak its
  exact wire format; you can bypass `format_multimodal_message` and
  hand-build the request.
- **Streaming raw binary out of a tool every turn.** If you really need
  bytes flowing without any serialization tax, talk to the LLM provider
  SDK directly for that path. The abstraction is built around
  request/response, not streams of bytes.

## Decision rubric

| Your situation                                                  | Recommendation                                  |
|-----------------------------------------------------------------|-------------------------------------------------|
| Single provider, no checkpoints, no tracing                     | Skip. Use raw base64 in your message-build code. |
| Two or more providers (or "we might switch")                    | Use the layer. Provider portability is its #1 job. |
| Durable workflows (Chain checkpoints, Swarm resume)             | Use the layer. Bytes don't checkpoint without it. |
| You ship the Local UI / Agent Replay / Eval dashboard           | Use the layer. The UI integrates around it.    |
| PDFs are involved at all                                        | Use the layer. The cross-provider PDF story is messy enough that hand-rolling is rarely worth it. |
| You're doing one-off prompt engineering against one model       | Skip. Two lines of `httpx.post` will do.       |

## See also

- [Images](images.md) — `Image` class, supported formats, sizes, the OpenAI `detail` parameter.
- [PDFs](pdfs.md) — `PDF` class, text vs vision vs native modes.
- [Provider Support](providers.md) — capability matrix and auto-detection rules.
- [Multimodal Eval](eval.md) — JSONL syntax for image/PDF test cases.
