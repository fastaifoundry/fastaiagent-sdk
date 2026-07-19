# Concepts & Mental Model

This page explains **what** the multimodal types are, **why** the SDK wraps
media in a type instead of letting you pass base64, and **the concept of how**
one `Image` becomes four different provider payloads — plus an explicit contract
for what survives a round-trip and what degrades to text. For task guides see
[Images](images.md), [PDFs](pdfs.md), [Files](files.md), and
[Providers](providers.md).

## What it is

`Image`, `PDF`, and `File` are first-class inputs you can pass to an agent
alongside text:

```python
agent.run(["What's wrong in this photo?", Image.from_file("damage.jpg")])
```

The same call works against OpenAI, Azure, Anthropic, Gemini, Bedrock, and
Ollama. A `ContentPart` is simply `str | Image | PDF | File`, and bare `bytes`
or a `Path` are auto-wrapped into a `File` at the boundary.

## Why a type, not a base64 string

The tempting shortcut is "just base64 it yourself." That breaks because
providers disagree on more than key names — **they disagree on the encoding
itself**:

- OpenAI wants a base64 **data URL** inline in the content array.
- Anthropic wants a base64 **source block**.
- Bedrock wants **raw bytes**, not base64 at all.
- Ollama wants images in a **sibling top-level array**, with the text flattened
  separately — not inline.

No single string can serve all four. So the SDK keeps the **canonical form as
raw bytes plus a mime type**, and derives the provider view at the edge. That's
the whole design: *bytes are the truth; every provider payload is a disposable
rendering of them.* (Encoding happens per formatting pass — it isn't cached —
so nothing holds a stale base64 copy.)

## The concept of how

### One branch point

There is a single function that knows provider wire formats for multimodal
messages, and every provider path funnels through it. It's pure — same inputs,
same output — which makes the per-provider differences testable in isolation
rather than smeared across the client.

!!! info "One honest exception"
    Gemini has its own builder, because everything there becomes `inlineData`
    and the PDF-mode planning below doesn't apply (Gemini reads PDFs directly
    and even extracts embedded text itself). So the abstraction has *two*
    implementations, not one.

### `pdf_mode` is a small planner

A PDF can reach a model three ways, and `pdf_mode="auto"` decides which:

| Mode | What happens |
|------|--------------|
| `native` | Send the PDF as a document part — the provider parses it |
| `vision` | Render pages to images locally and send those |
| `text` | Extract text locally and send that |

`auto` resolves in order: **native** if the capability registry says this model
parses PDFs, else **vision** if the model has vision, else **text**. The
important subtlety: it's the resolved *plan*, not the input type, that decides
whether a vision-capable model is required — a PDF going out in `text` mode
needs no vision at all.

The registry that answers "is this model vision-capable / can it read PDFs
natively" is **prefix-matched** on model names, with explicit overrides for
misleading prefixes (a bare `gpt-4` is not a vision model).

### Guardrails against the physics of media

Because media is big and comes from the outside world, several protections are
built in rather than left to you:

- **Per-provider size ceilings** deliberately set *below* the published limits,
  to leave headroom for base64's ~33% inflation.
- **Auto-resize instead of failing** — an oversized image is stepped down 25% at
  a time (floor 256px), JPEG for photos and PNG for lossless so diagrams don't
  pick up artifacts, with a warning each time.
- **Decompression-bomb defence** — the pixel ceiling is lowered and the usual
  warning is promoted to an error, so it fails closed.
- **SSRF hardening on every `from_url`** — HTTP(S) only, private/loopback/
  link-local hosts refused unless explicitly allowed, with timeout, redirect,
  and size caps.

### Tool returns serve two audiences at once

When a tool returns an image, the SDK produces **two** things from that single
return: the real `ContentPart` list that flows back to the model verbatim, and a
text summary used for tracing, guardrails, and memory. The model gets the bytes;
the operator gets something readable. It's the cleanest illustration of the
whole design.

## What survives, and what degrades

This is the contract worth internalizing — the SDK is deliberately not uniform
here, because not every consumer can hold bytes:

| Tier | Where | Behavior |
|------|-------|----------|
| **Full fidelity** | Chain state & checkpoints, tool→LLM returns, trace attachments | Bytes preserved exactly |
| **Degrades to a text marker** | Memory, guardrails, span attributes | Becomes e.g. `[image:image/jpeg:20481b]` |
| **Not handled** | `File` in chain checkpoints and in the input summary | See the warning below |

Memory backends, guardrails, and span attributes are **text-only**, so a
multimodal input is recorded as a compact marker rather than the payload. The
bytes aren't lost to observability though — they're persisted separately as
trace attachments so the UI can render them.

!!! info "Verified against a live run"
    `examples/45_multimodal_chain.py` put an `Image` and a `PDF` into chain
    state, checkpointed, and simulated a process restart. Both serialized into
    the snapshot as typed dicts (`type: "image"` / `"pdf"`), rehydrated as real
    `Image` / `PDF` objects, and compared **byte-identical** to the originals.

!!! warning "`File` has gaps in the round-trip"
    The checkpoint walker and the input-summary helper handle `Image` and `PDF`
    but not `File` — even though `File.to_dict()`/`from_dict()` exist. A `File`
    placed in chain state will not serialize cleanly, and a `File` input
    contributes nothing to the text summary used for memory/guardrails/spans.
    Prefer `Image`/`PDF` for anything that must survive a checkpoint, and pass
    `File` as direct agent input.

## Imports

`fastaiagent` exports `Image`, `PDF`, `File`, and `ContentPart`. The lower-level
helpers (`format_multimodal_message`, `is_vision_capable`, `supports_native_pdf`,
`maybe_resize`) live in `fastaiagent.multimodal` — a deliberate public/internal
seam.

## Next steps

- [Multimodal overview](index.md) — the unified call and support matrix
- [When to use it](when-to-use.md) — the abstraction vs. hand-rolling base64
- [Images](images.md) · [PDFs](pdfs.md) · [Files](files.md)
- [Providers](providers.md) — per-provider wire-format support
- [Multimodal eval](eval.md) — typed parts in datasets
