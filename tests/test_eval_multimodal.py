"""Phase 6 tests — Eval JSONL loader recognizes image/pdf parts.

Mock-free: real JSONL on disk, real ``Dataset.from_jsonl`` reading real
fixture binaries via real Pillow / pymupdf. No HTTP and no mock LLM.

Spec test #12.
"""

from __future__ import annotations

from pathlib import Path

from fastaiagent import PDF, Dataset, Image

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"


def test_from_jsonl_passes_string_inputs_through_unchanged() -> None:
    ds = Dataset.from_jsonl(FIXTURES / "eval_dataset.jsonl")
    items = list(ds)
    assert items[0]["input"] == "plain text input"
    assert items[0]["expected"] == "ok"


def test_from_jsonl_resolves_image_path_to_image_instance() -> None:
    ds = Dataset.from_jsonl(FIXTURES / "eval_dataset.jsonl")
    items = list(ds)
    parts = items[1]["input"]
    assert isinstance(parts, list)
    assert parts[0] == "What letters do you see?"
    assert isinstance(parts[1], Image)
    assert parts[1].media_type == "image/jpeg"
    assert parts[1].size_bytes() > 0


def test_from_jsonl_resolves_pdf_path_to_pdf_instance() -> None:
    ds = Dataset.from_jsonl(FIXTURES / "eval_dataset.jsonl")
    items = list(ds)
    parts = items[2]["input"]
    assert isinstance(parts, list)
    assert isinstance(parts[1], PDF)
    assert parts[1].page_count() == 2
    assert "Service Agreement" in parts[1].extract_text()


def test_from_jsonl_passes_image_detail_through_to_constructor() -> None:
    ds = Dataset.from_jsonl(FIXTURES / "eval_dataset.jsonl")
    items = list(ds)
    parts = items[3]["input"]
    assert isinstance(parts[1], Image)
    assert parts[1].detail == "high"
    assert parts[1].media_type == "image/png"


def test_paths_resolve_relative_to_jsonl_directory(tmp_path: Path) -> None:
    """A user can move the dataset file anywhere; relative paths must
    follow the JSONL — not the cwd at load time."""
    nested = tmp_path / "nested" / "dataset.jsonl"
    nested.parent.mkdir()
    # Copy fixtures into the nested dir so we exercise relative resolution.
    cat_dest = nested.parent / "cat.jpg"
    cat_dest.write_bytes((FIXTURES / "cat.jpg").read_bytes())
    nested.write_text(
        '{"input": [{"type":"text","text":"x"},{"type":"image","path":"cat.jpg"}], '
        '"expected":"x"}\n'
    )

    ds = Dataset.from_jsonl(nested)
    parts = list(ds)[0]["input"]
    assert isinstance(parts[1], Image)
    assert parts[1].size_bytes() == cat_dest.stat().st_size


def test_dataset_can_be_iterated_for_evaluate_loop() -> None:
    """The ``input`` field must be in the shape ``Agent.run`` accepts —
    a string or a list of ``str | Image | PDF``."""
    ds = Dataset.from_jsonl(FIXTURES / "eval_dataset.jsonl")
    for item in ds:
        inp = item["input"]
        if isinstance(inp, str):
            continue
        assert isinstance(inp, list)
        for p in inp:
            assert isinstance(p, (str, Image, PDF))
