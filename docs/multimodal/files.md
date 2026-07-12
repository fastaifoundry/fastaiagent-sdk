# Files (any type)

`fastaiagent.File` is the generic native file input. Hand it any bytes and a
mime type (sniffed automatically) and the SDK forwards it to the model using
each provider's **native** file mechanism — never stringified, never rendered
locally.

```python
from fastaiagent import Agent, File

agent.run(["Extract the totals.", File.from_bytes(file_bytes)])
```

Even simpler — bare bytes or a path are auto-detected and wrapped for you:

```python
agent.run(file_bytes)            # mime sniffed from the bytes
agent.run(Path("report.pdf"))    # mime from content + extension
agent.run(["Summarize this.", Path("data.csv")])
```

## Constructors

```python
File.from_bytes(data, mime_type=None, filename=None)   # sniffs mime if omitted
File.from_path("report.docx")                          # mime from content/ext
File.from_url("https://example.com/report.pdf")        # SSRF-hardened, 100 MiB cap
File.from_file_id("file-abc123", mime_type="application/pdf")  # provider-uploaded
```

`File` carries `data`, `mime_type`, `filename`, and an optional `file_id` (for
files already uploaded to a provider's Files API). Its `.category` is one of
`image`, `audio`, `video`, `pdf`, `document`, or `other`.

## What each provider accepts natively

The formatter routes by mime type. Where a provider has no native path for a
type, the SDK raises a clear `MultimodalError` with the fallback (upload for a
`file_id`, or extract text) rather than silently degrading.

| Category | OpenAI / Azure | Anthropic | Gemini | Bedrock |
|----------|----------------|-----------|--------|---------|
| image    | `image_url`    | image block | `inlineData` | image block |
| audio (wav/mp3) | `input_audio` | — | `inlineData` | — |
| pdf      | `file` (base64) | `document` | `inlineData` | `document` |
| document (csv/txt/html/md) | `file_id` only¹ | text `document` | `inlineData` | `document` |
| office (docx/xlsx) | `file_id` only¹ | `file_id` only | — | `document` |

¹ OpenAI/Azure Chat Completions take **only PDF** as inline `file_data`; other
document types must be uploaded first (Files API) and passed via
`File.from_file_id(...)`.

## `Image` and `PDF` still work

`File` sits alongside the existing `Image` and `PDF` inputs — nothing changes
for them. `PDF` remains the right choice when you want `pdf_mode="text"` /
`"vision"` control; `File` always forwards natively.
