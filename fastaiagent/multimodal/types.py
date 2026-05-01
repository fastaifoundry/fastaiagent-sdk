"""``ContentPart`` and ``normalize_input`` — the unified shape for multimodal inputs."""

from __future__ import annotations

from fastaiagent.multimodal.image import Image
from fastaiagent.multimodal.pdf import PDF

ContentPart = str | Image | PDF


def normalize_input(
    input: str | Image | PDF | list[ContentPart],
) -> list[ContentPart]:
    """Coerce any accepted ``Agent.run`` input into ``list[ContentPart]``.

    Accepted shapes:

    * ``str`` — wrapped in a single-element list (backward compatible)
    * single ``Image`` or ``PDF`` — wrapped in a single-element list
    * ``list`` whose items are each ``str``, ``Image``, or ``PDF`` — copied

    Anything else raises ``TypeError``. We deliberately do not accept bare
    ``bytes`` (ambiguous: image bytes? pdf bytes?) — callers must wrap them
    using ``Image.from_bytes`` / ``PDF.from_bytes``.
    """
    if isinstance(input, str):
        return [input]
    if isinstance(input, (Image, PDF)):
        return [input]
    if isinstance(input, list):
        for i, part in enumerate(input):
            if not isinstance(part, (str, Image, PDF)):
                raise TypeError(
                    f"input[{i}] has unsupported type {type(part).__name__!r}; "
                    f"each part must be str, Image, or PDF"
                )
        return list(input)
    raise TypeError(
        f"unsupported input type {type(input).__name__!r}; "
        f"expected str | Image | PDF | list[ContentPart]"
    )
