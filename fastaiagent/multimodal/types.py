"""``ContentPart`` and ``normalize_input`` — the unified shape for multimodal inputs."""

from __future__ import annotations

import os
from pathlib import Path

from fastaiagent.multimodal.file import File
from fastaiagent.multimodal.image import Image
from fastaiagent.multimodal.pdf import PDF

ContentPart = str | Image | PDF | File

# What a caller may hand a single part as (before normalization). ``bytes`` and
# ``os.PathLike`` are auto-detected and wrapped in ``File`` (mime sniffed).
RawPart = str | Image | PDF | File | bytes | os.PathLike[str]


def _coerce_part(part: object, idx: int) -> ContentPart:
    if isinstance(part, (str, Image, PDF, File)):
        return part
    if isinstance(part, (bytes, bytearray)):
        return File.from_bytes(bytes(part))
    if isinstance(part, os.PathLike) or isinstance(part, Path):
        return File.from_path(part)  # type: ignore[arg-type]
    raise TypeError(
        f"input[{idx}] has unsupported type {type(part).__name__!r}; "
        f"each part must be str, Image, PDF, File, bytes, or a path"
    )


def normalize_input(
    input: RawPart | list[RawPart],
) -> list[ContentPart]:
    """Coerce any accepted ``Agent.run`` input into ``list[ContentPart]``.

    Accepted shapes:

    * ``str`` — wrapped in a single-element list (backward compatible)
    * single ``Image`` / ``PDF`` / ``File`` — wrapped in a single-element list
    * ``bytes`` or a path (``os.PathLike``) — mime sniffed and wrapped in
      ``File`` (so ``agent.run(file_bytes)`` / ``agent.run(Path("x.pdf"))`` work)
    * ``list`` whose items are each of the above — coerced element-wise
    """
    if isinstance(input, str):
        return [input]
    if isinstance(input, list):
        return [_coerce_part(part, i) for i, part in enumerate(input)]
    # Single non-list part (str handled above).
    return [_coerce_part(input, 0)]
