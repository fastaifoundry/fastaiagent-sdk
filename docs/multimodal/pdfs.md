# PDFs

`fastaiagent.PDF` carries a PDF document into an LLM call. Three
constructors:

```python
from fastaiagent import PDF

pdf = PDF.from_file("contract.pdf")
pdf = PDF.from_bytes(raw)
pdf = PDF.from_url("https://example.com/report.pdf")
```

`PDF.from_url` is SSRF-hardened: only public `http(s)` hosts are
accepted, every redirect hop is re-validated, and the body is capped at
100 MiB. Set `FASTAIAGENT_ALLOW_PRIVATE_NETWORKS=1` to opt into intranet
fetching. See [Image URL safety](images.md#url-safety) for the full ruleset.

## Processing modes

`pdf_mode` controls how the PDF reaches the LLM:

| Mode      | Wire format                                   | Cost  | Layout fidelity |
|-----------|-----------------------------------------------|-------|-----------------|
| `text`    | `pymupdf` extracts text → single text block   | Low   | None — bare text |
| `vision`  | Pages rendered to PNG → image blocks per page | High  | High — preserves tables, signatures |
| `native`  | Raw PDF forwarded to the provider (Anthropic `document` block / OpenAI `file` part) | Med | Highest — the model reads the PDF directly |
| `auto`    | Default — picks the best mode for the model   | Mixed | Mixed |

### auto resolution

* Anthropic Sonnet/Opus 3.5+ → `native` (one document block; lowest cost)
* OpenAI/Azure vision models (gpt-4o, gpt-4.1, gpt-5, o-series) → `native`
  (raw PDF forwarded as a `file` part; the provider parses it server-side)
* Bedrock-hosted Claude → `native` (Converse `document` block)
* Gemini → always native (`inlineData`); `pdf_mode` does not apply
* Any other vision-capable model (Ollama, Mistral, custom) → `vision`
* Non-vision model (gpt-3.5-turbo, claude-2.1, …) → `text`

`native` mode forwards the **whole** PDF — it does not render or extract
locally, so `max_pdf_pages` does not apply and PDFs that `pymupdf` cannot
decompress (e.g. some flate-compressed streams) still work. Custom
OpenAI-compatible endpoints stay on `vision` under `auto`; pass
`pdf_mode="native"` explicitly if your endpoint accepts the `file` part.

Configure globally or per-LLMClient:

```python
import fastaiagent as fa
fa.config.pdf_mode = "vision"          # default "auto"

LLMClient(provider="openai", model="gpt-4o", pdf_mode="vision")
```

## Page limit

Vision mode caps pages by default to keep token costs bounded:

```python
fa.config.max_pdf_pages = 20            # default 20

# Or per-LLMClient:
LLMClient(provider="openai", model="gpt-4o", max_pdf_pages=50)
```

When a PDF exceeds the limit, the extra pages are dropped and a warning
is logged. For very long documents prefer `pdf_mode="text"` or build a
two-stage pipeline (chunk → summarise → vision pass on the relevant pages).

## Extract text directly

`PDF.extract_text()` returns the joined text of every page, useful when
chunking before a RAG pipeline:

```python
pdf = PDF.from_file("contract.pdf")
text = pdf.extract_text()
print(pdf.page_count(), "pages,", len(text), "chars")
```

## Render pages to images

`PDF.to_page_images(dpi=150, max_pages=None)` returns a `list[Image]`. The
SDK calls this for `pdf_mode="vision"`; you rarely need it directly.

```python
pages = pdf.to_page_images(dpi=200, max_pages=5)
for i, page in enumerate(pages):
    page.to_dict()   # serializable Image
```

## Costs and latency

PDF rendering at 150 dpi via pymupdf takes ~200–400 ms per page on a
modern laptop. Adding 20 page images to a single chat request adds
roughly 20× a single-image vision call's tokens — read your provider's
pricing page before turning vision mode on by default.
