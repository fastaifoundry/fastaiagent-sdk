"""Bring-your-own-parser path for ``PDF`` (1.56.0).

Local PDF decoding is optional. A caller who already parses PDFs with their own
library — pdfplumber, pypdf, Tika, a vendor OCR API — hands the SDK the result
via ``PDF(text=...)`` and never touches the SDK's engine.

Real objects throughout, no mocks. The "no engine installed" case is exercised
by hiding ``pypdfium2`` from ``importlib.util.find_spec`` for the duration of a
single call, which is how ``_pdf_backend.available()`` decides. That is a real
absence of a real module, not a stubbed return value; the end-to-end version of
the same assertion runs against a genuinely engine-free venv in
``scripts/check_core_surface.py`` under the ``clean-core`` CI job.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from fastaiagent import PDF
from fastaiagent._internal.errors import MissingPDFBackendError, MultimodalError

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"
CONTRACT = FIXTURES / "contract.pdf"

BYO_TEXT = "Page 1: parsed by somebody else's library.\n\nPage 2: still theirs."


@contextmanager
def no_pdf_engine(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make the PDF engine genuinely unfindable, as on a core-only install."""
    real = importlib.util.find_spec

    def hidden(name: str, package: str | None = None):  # type: ignore[no-untyped-def]
        if name == "pypdfium2":
            return None
        return real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", hidden)
    yield


# --- BYO text is returned verbatim, engine or not ---


def test_extract_text_returns_supplied_text() -> None:
    pdf = PDF.from_file(CONTRACT, text=BYO_TEXT)
    assert pdf.extract_text() == BYO_TEXT


def test_supplied_text_takes_precedence_over_the_engine() -> None:
    """The engine IS installed here; the caller's text must still win."""
    pdf = PDF.from_file(CONTRACT, text=BYO_TEXT)
    assert "Service Agreement" in PDF.from_file(CONTRACT).extract_text()
    assert pdf.extract_text() == BYO_TEXT
    assert "Service Agreement" not in pdf.extract_text()


def test_byo_text_works_with_no_engine_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = PDF.from_file(CONTRACT, text=BYO_TEXT)
    with no_pdf_engine(monkeypatch):
        assert pdf.extract_text() == BYO_TEXT


def test_from_bytes_accepts_text() -> None:
    pdf = PDF.from_bytes(CONTRACT.read_bytes(), text=BYO_TEXT)
    assert pdf.extract_text() == BYO_TEXT


def test_all_three_constructors_expose_text_as_keyword_only() -> None:
    """``text`` must be keyword-only on every constructor, so adding it cannot
    shift the meaning of any existing positional argument."""
    import inspect

    for ctor in (PDF.from_file, PDF.from_bytes, PDF.from_url):
        param = inspect.signature(ctor).parameters.get("text")
        assert param is not None, f"{ctor.__name__} is missing text="
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{ctor.__name__}: text= must be keyword-only"
        )
        assert param.default is None


def test_text_defaults_to_none_and_does_not_change_existing_behaviour() -> None:
    pdf = PDF.from_file(CONTRACT)
    assert pdf.text is None
    assert "Service Agreement" in pdf.extract_text()


# --- serialization round-trip (checkpoint / resume) ---


def test_to_dict_omits_text_when_absent() -> None:
    """Payloads for the common case stay byte-identical to pre-1.56.0 ones."""
    assert "text" not in PDF.from_file(CONTRACT).to_dict()


def test_to_dict_from_dict_round_trips_supplied_text() -> None:
    pdf = PDF.from_file(CONTRACT, text=BYO_TEXT)
    restored = PDF.from_dict(pdf.to_dict())
    assert restored.text == BYO_TEXT
    assert restored.extract_text() == BYO_TEXT
    assert restored.data == pdf.data


def test_from_dict_accepts_payloads_written_before_this_field_existed() -> None:
    legacy = {
        "type": "pdf",
        "data_base64": PDF.from_file(CONTRACT).to_base64(),
        "source_path": "contract.pdf",
        "source_url": None,
    }
    assert PDF.from_dict(legacy).text is None


def test_survives_a_chain_checkpoint_round_trip() -> None:
    from fastaiagent.chain.state import _hydrate_from_checkpoint, _serialize_for_checkpoint

    pdf = PDF.from_file(CONTRACT, text=BYO_TEXT)
    restored = _hydrate_from_checkpoint(_serialize_for_checkpoint({"doc": pdf}))["doc"]
    assert isinstance(restored, PDF)
    assert restored.extract_text() == BYO_TEXT


# --- the error raised when there is no engine and no supplied text ---


def test_missing_engine_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = PDF.from_file(CONTRACT)
    with no_pdf_engine(monkeypatch):
        with pytest.raises(MissingPDFBackendError) as exc:
            pdf.extract_text()
    message = str(exc.value)
    assert 'fastaiagent[pdf]' in message, "must name the extra"
    assert "text=" in message, "must name the bring-your-own escape hatch"
    assert 'pdf_mode="native"' in message, "must name the no-engine-needed mode"


def test_missing_engine_error_is_still_catchable_as_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compatibility guarantee for this release.

    Before the ``pdf`` extra existed this path raised ``ModuleNotFoundError``
    from ``import pymupdf``. Anyone catching ``ImportError`` must keep working.
    """
    pdf = PDF.from_file(CONTRACT)
    with no_pdf_engine(monkeypatch):
        with pytest.raises(ImportError):
            pdf.extract_text()
        with pytest.raises(MultimodalError):
            pdf.page_count()
        with pytest.raises(ImportError):
            pdf.to_page_images()


@pytest.mark.parametrize("method", ["page_count", "to_page_images"])
def test_supplied_text_does_not_satisfy_the_rendering_methods(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``text=`` covers extraction only — page geometry still needs an engine."""
    pdf = PDF.from_file(CONTRACT, text=BYO_TEXT)
    with no_pdf_engine(monkeypatch):
        with pytest.raises(MissingPDFBackendError):
            getattr(pdf, method)()
