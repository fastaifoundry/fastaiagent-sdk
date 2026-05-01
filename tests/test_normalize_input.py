"""Tests for ``fastaiagent.multimodal.normalize_input``."""

from __future__ import annotations

from pathlib import Path

import pytest

from fastaiagent import PDF, Image, normalize_input

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"


def test_string_wrapped_in_list() -> None:
    assert normalize_input("hello") == ["hello"]


def test_single_image_wrapped_in_list() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    assert normalize_input(img) == [img]


def test_single_pdf_wrapped_in_list() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    assert normalize_input(pdf) == [pdf]


def test_mixed_list_preserves_order() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    result = normalize_input(["caption", img, "between", pdf])
    assert len(result) == 4
    assert result[0] == "caption"
    assert result[1] is img
    assert result[2] == "between"
    assert result[3] is pdf


def test_empty_list_returns_empty_list() -> None:
    assert normalize_input([]) == []


def test_returns_a_copy_of_list_not_alias() -> None:
    parts: list[str] = ["a", "b"]
    result = normalize_input(parts)
    assert result == parts
    assert result is not parts


def test_bare_bytes_rejected() -> None:
    with pytest.raises(TypeError):
        normalize_input(b"raw bytes")  # type: ignore[arg-type]


def test_dict_rejected() -> None:
    with pytest.raises(TypeError):
        normalize_input({"text": "hi"})  # type: ignore[arg-type]


def test_list_with_invalid_element_rejected() -> None:
    with pytest.raises(TypeError, match=r"input\[1\]"):
        normalize_input(["ok", 42])  # type: ignore[list-item]
