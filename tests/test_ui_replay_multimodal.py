"""Phase 7 backend tests — REST modify-fork accepts multimodal input.

The frontend's "Replace image" button posts a JSON list with base64 data
URLs; the backend resolver turns those into real ``Image``/``PDF``
instances before handing them to ``forked.modify_input``. These tests
exercise the resolver directly with real Pillow bytes — no HTTP, no mocks.
"""

from __future__ import annotations

import base64
from pathlib import Path

from fastaiagent import PDF, Image
from fastaiagent.ui.routes.replay import _resolve_modify_input

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"


def _b64_of(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")


def test_resolves_text_part_to_string() -> None:
    out = _resolve_modify_input([{"type": "text", "text": "hello"}])
    assert out == ["hello"]


def test_resolves_image_part_to_image_instance() -> None:
    cat_b64 = _b64_of(FIXTURES / "cat.jpg")
    out = _resolve_modify_input(
        [
            {"type": "text", "text": "describe"},
            {"type": "image", "data_base64": cat_b64, "media_type": "image/jpeg"},
        ]
    )
    assert out[0] == "describe"
    assert isinstance(out[1], Image)
    assert out[1].media_type == "image/jpeg"
    assert out[1].size_bytes() > 0


def test_resolves_pdf_part_to_pdf_instance() -> None:
    pdf_b64 = _b64_of(FIXTURES / "contract.pdf")
    out = _resolve_modify_input([{"type": "pdf", "data_base64": pdf_b64}])
    assert isinstance(out[0], PDF)
    assert out[0].page_count() == 2


def test_passes_strings_through_in_list() -> None:
    out = _resolve_modify_input(["a", "b"])
    assert out == ["a", "b"]


def test_legacy_string_input_unchanged() -> None:
    assert _resolve_modify_input("plain text") == "plain text"


def test_legacy_dict_input_unchanged() -> None:
    assert _resolve_modify_input({"input": "from dict"}) == {"input": "from dict"}


def test_passes_image_detail_through() -> None:
    cat_b64 = _b64_of(FIXTURES / "cat.jpg")
    out = _resolve_modify_input(
        [{"type": "image", "data_base64": cat_b64, "media_type": "image/jpeg", "detail": "high"}]
    )
    assert out[0].detail == "high"
