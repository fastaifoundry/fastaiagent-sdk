"""Multimodal input types — Image and PDF.

These are first-class inputs to ``Agent.run``, ``Chain.execute``, ``Swarm.run``
and friends. Provider-specific wire formatting happens later, inside
``LLMClient`` (see ``fastaiagent.multimodal.format``).
"""

from fastaiagent.multimodal.format import format_multimodal_message
from fastaiagent.multimodal.image import Image
from fastaiagent.multimodal.pdf import PDF
from fastaiagent.multimodal.registry import is_vision_capable, supports_native_pdf
from fastaiagent.multimodal.resize import maybe_resize
from fastaiagent.multimodal.types import ContentPart, normalize_input

__all__ = [
    "Image",
    "PDF",
    "ContentPart",
    "normalize_input",
    "format_multimodal_message",
    "is_vision_capable",
    "supports_native_pdf",
    "maybe_resize",
]
