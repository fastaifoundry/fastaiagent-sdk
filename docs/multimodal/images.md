# Images

`fastaiagent.Image` represents a single image input to an Agent, Chain,
Swarm, or Supervisor. Three constructors:

```python
from fastaiagent import Image

# 1. Local file — Pillow sniffs the media_type from content
img = Image.from_file("photo.jpg")

# 2. URL — fetched over HTTPS only, max 5 redirects, 30 s timeout
img = Image.from_url("https://example.com/photo.png")

# 3. Raw bytes — explicit media_type required
img = Image.from_bytes(raw, media_type="image/jpeg")
```

## Supported formats

| Format | media_type     | Notes                       |
|--------|----------------|-----------------------------|
| JPEG   | `image/jpeg`   | Most efficient for photos   |
| PNG    | `image/png`    | Best for screenshots / UIs  |
| GIF    | `image/gif`    | First-frame only on most providers |
| WebP   | `image/webp`   | Requires libwebp on Linux   |

BMP, SVG, TIFF, and HEIC are rejected with `UnsupportedFormatError`.
Convert to JPEG/PNG before passing in.

## Detail parameter (OpenAI)

OpenAI's vision models accept a `detail` hint that controls token cost:

```python
img = Image.from_file("photo.jpg", detail="high")  # full-resolution vision pass
img = Image.from_file("photo.jpg", detail="low")   # 512 px / 85 tokens flat
img = Image.from_file("photo.jpg", detail="auto")  # default — provider chooses
```

`detail` is silently ignored by providers that do not support it.

## Auto-resize

Vision endpoints have per-image size limits (OpenAI 20 MB, Anthropic 5 MB,
Bedrock 5 MB). When an image exceeds the cap the SDK resizes it on the fly
using Pillow's `thumbnail()` and logs a warning:

```
WARNING fastaiagent.multimodal.resize: auto-resized image from 23000 KB
(JPEG) to 4500 KB (image/jpeg) to fit 4.5 MB limit
```

Override the cap globally:

```python
import fastaiagent as fa
fa.config.max_image_size_mb = 10.0
```

…or per-LLMClient:

```python
LLMClient(provider="openai", model="gpt-4o", max_image_size_mb=8.0)
```

## URL safety

`Image.from_url` is SSRF-hardened:

* Only `http(s)` schemes are accepted (`file://`, `data:`, etc. are rejected).
* The host must resolve to a **public** address. Private RFC 1918
  (`10/8`, `172.16/12`, `192.168/16`), loopback (`127/8`, `::1`),
  link-local (`169.254/16` — cloud metadata service),
  reserved/multicast/unspecified ranges are refused.
* Each redirect hop is re-validated against the same rules — an attacker
  cannot 302-bounce a public URL into the internal network.
* Body is capped at 25 MiB; redirects at 5; timeout at 30s.

Set `FASTAIAGENT_ALLOW_PRIVATE_NETWORKS=1` in the environment to opt into
intranet fetching (e.g. an internal image server). When the bytes are
already in memory, prefer `Image.from_bytes`.

## Serialization

`Image` is dataclass-equal by value and survives JSON round-trips via
`to_dict()` / `from_dict()`:

```python
encoded = img.to_dict()                         # {"type": "image", ...}
restored = Image.from_dict(encoded)
assert restored == img
```

This is what makes `Image` work inside [Chain state](../chains/index.md)
and the Replay fork dialog.
