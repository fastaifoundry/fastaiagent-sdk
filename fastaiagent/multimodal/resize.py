"""Auto-resize oversized images before sending to a provider.

Each vendor caps individual image payloads (OpenAI: 20 MB, Anthropic: 5 MB,
Bedrock/Vertex: similar). Rather than fail at the API boundary, the SDK
detects oversize inputs and re-encodes them at a lower resolution so the
request succeeds. A warning is logged so the developer notices.
"""

from __future__ import annotations

import io
import logging

from fastaiagent.multimodal.image import Image

logger = logging.getLogger(__name__)

# Default ceiling matches OpenAI's 20 MB per-image limit. Other providers cap
# lower; callers pass a tighter ``max_mb`` for those.
DEFAULT_MAX_IMAGE_MB: float = 20.0

# Pillow's resize uses 1-pixel chunking; sub-1024 images are rarely worth
# resizing. We refuse to scale below this regardless of ``max_mb``.
_MIN_DIMENSION: int = 256


def maybe_resize(image: Image, *, max_mb: float = DEFAULT_MAX_IMAGE_MB) -> Image:
    """Return a smaller :class:`Image` if ``image`` exceeds ``max_mb``, else ``image``.

    The original is never mutated. Aspect ratio is preserved. Output uses
    JPEG (q=85) for lossy formats and PNG for lossless GIFs/WebP/PNGs to
    avoid artifacts on diagrams. A warning is logged on every resize.
    """
    threshold_bytes = int(max_mb * 1024 * 1024)
    if image.size_bytes() <= threshold_bytes:
        return image

    from PIL import Image as PILImage

    with PILImage.open(io.BytesIO(image.data)) as pil:
        pil.load()
        original_format = (pil.format or "").upper()
        # Walk down by 25% steps until under budget or below the floor.
        target = pil.copy()
        scale = 1.0
        while True:
            buf = io.BytesIO()
            save_format = "JPEG" if image.media_type == "image/jpeg" else "PNG"
            if save_format == "JPEG":
                if target.mode != "RGB":
                    target = target.convert("RGB")
                target.save(buf, format="JPEG", quality=85)
            else:
                target.save(buf, format="PNG")
            new_bytes = buf.getvalue()
            if len(new_bytes) <= threshold_bytes:
                break
            scale *= 0.75
            new_w = int(pil.width * scale)
            new_h = int(pil.height * scale)
            if new_w < _MIN_DIMENSION or new_h < _MIN_DIMENSION:
                break
            target = pil.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

    new_media_type = "image/jpeg" if save_format == "JPEG" else "image/png"
    logger.warning(
        "auto-resized image from %d KB (%s) to %d KB (%s) to fit %.1f MB limit",
        image.size_bytes() // 1024,
        original_format or image.media_type,
        len(new_bytes) // 1024,
        new_media_type,
        max_mb,
    )
    return Image(
        data=new_bytes,
        media_type=new_media_type,
        source_url=image.source_url,
        detail=image.detail,
    )
