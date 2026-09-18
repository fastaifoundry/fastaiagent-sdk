"""``fastaiagent.config`` is a real, readable, *acted-upon* settings surface.

Three user-facing surfaces told people to set ``fa.config.<x>`` — including a
live HTTP 404 body served by the local UI — while ``fastaiagent.config`` did not
exist at all, and three of the settings it pointed at (``pdf_mode``,
``max_pdf_pages``, ``max_image_size_mb``) had no path to any code because
``multimodal/format.py`` never called ``get_config()``.

The rule this file enforces: **never expose a setter that nothing reads.** A
silently-ignored setting is strictly worse than an honest ``AttributeError``, so
every field reachable through ``fa.config`` gets a behavioural round-trip here,
not just an attribute check.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import fastaiagent as fa
from fastaiagent._internal.config import SDKConfig, get_config, reset_config

_REPO_ROOT = Path(__file__).resolve().parent.parent

# A 1x1 PNG — small enough that no cap trims it, real enough for the encoders.
PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6360000002000100ffff03000006000557bfabd4000000"
    "0049454e44ae426082"
)


@pytest.fixture(autouse=True)
def _fresh_config():
    reset_config()
    yield
    reset_config()


def test_fa_config_is_the_singleton():
    assert fa.config is get_config()
    assert isinstance(fa.config, SDKConfig)
    assert "config" in fa.__all__


def test_assignment_sticks_because_the_model_is_mutable():
    fa.config.log_level = "DEBUG"
    assert get_config().log_level == "DEBUG"


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        fa.not_a_real_attribute  # noqa: B018


# ── trace_full_images actually gates full_data ─────────────────────────────


def _save_one_image(db):
    from fastaiagent.multimodal.image import Image
    from fastaiagent.trace.attachments import save_parts_for_span

    return save_parts_for_span(
        db=db,
        trace_id="a" * 32,
        span_id="b" * 16,
        parts=[Image(data=PNG_1PX, media_type="image/png")],
        role="input",
    )


@pytest.fixture
def trace_db(tmp_path, monkeypatch):
    from fastaiagent.ui.db import init_local_db

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    db = init_local_db(str(tmp_path / "local.db"))
    yield db
    db.close()


def test_trace_full_images_false_stores_thumbnail_only(trace_db):
    fa.config.trace_full_images = False
    saved = _save_one_image(trace_db)
    assert saved, "nothing was persisted — the round-trip proves nothing"
    row = trace_db.fetchone(
        "SELECT size_bytes, full_data FROM trace_attachments WHERE attachment_id = ?",
        (saved[0].attachment_id,),
    )
    # The row is written either way — what the setting gates is the originals.
    assert row["size_bytes"] == len(PNG_1PX)
    assert row["full_data"] is None


def test_trace_full_images_true_stores_the_original_bytes(trace_db):
    fa.config.trace_full_images = True
    saved = _save_one_image(trace_db)
    row = trace_db.fetchone(
        "SELECT full_data FROM trace_attachments WHERE attachment_id = ?",
        (saved[0].attachment_id,),
    )
    assert row["full_data"] == PNG_1PX


def test_trace_full_images_reads_from_the_environment(monkeypatch):
    monkeypatch.setenv("FASTAIAGENT_TRACE_FULL_IMAGES", "1")
    reset_config()
    assert fa.config.trace_full_images is True
    monkeypatch.setenv("FASTAIAGENT_TRACE_FULL_IMAGES", "off")
    reset_config()
    assert fa.config.trace_full_images is False


# ── the three multimodal globals reach format_multimodal_message ───────────


def _client(**kwargs):
    from fastaiagent.llm.client import LLMClient

    kwargs.setdefault("provider", "anthropic")
    kwargs.setdefault("model", "claude-sonnet-4-6")
    kwargs.setdefault("api_key", "k")
    return LLMClient(**kwargs)


def test_pdf_mode_global_reaches_the_emitted_blocks():
    """``fa.config.pdf_mode = "text"`` must change the block shape, not just a field."""
    from fastaiagent.multimodal.format import format_multimodal_message
    from fastaiagent.multimodal.pdf import PDF

    pdf = PDF(data=_tiny_pdf())

    fa.config.pdf_mode = "native"
    native = format_multimodal_message([pdf], "anthropic", **_mm_kwargs(_client()))
    assert any(b.get("type") == "document" for b in native["content"]), native

    fa.config.pdf_mode = "text"
    textual = format_multimodal_message([pdf], "anthropic", **_mm_kwargs(_client()))
    assert all(b.get("type") == "text" for b in textual["content"]), textual


def test_an_explicit_kwarg_beats_the_global():
    fa.config.pdf_mode = "native"
    assert _client(pdf_mode="text").pdf_mode == "text"
    assert _client().pdf_mode == "native"


def test_max_pdf_pages_global_reaches_the_client():
    fa.config.max_pdf_pages = 3
    assert _client().max_pdf_pages == 3
    assert _client(max_pdf_pages=7).max_pdf_pages == 7


def test_max_image_size_mb_default_keeps_the_provider_limit():
    """The trap: ``None`` means "use the provider's own cap".

    ``SDKConfig.max_image_size_mb`` therefore defaults to ``None`` and not to a
    number — applying a blanket 20 MB unconditionally would have *raised*
    Anthropic's effective per-image ceiling from 5 MB to 20 MB, which is a
    regression dressed as a feature.
    """
    from fastaiagent.multimodal.format import _PROVIDER_IMAGE_LIMIT_MB

    assert SDKConfig().max_image_size_mb is None
    assert _client().max_image_size_mb is None
    assert _PROVIDER_IMAGE_LIMIT_MB.get("anthropic", 0) < 20.0


def test_max_image_size_mb_global_overrides_when_the_operator_sets_it():
    fa.config.max_image_size_mb = 9.0
    assert _client().max_image_size_mb == 9.0
    # An explicit None still means "provider limit" and must win over the global.
    assert _client(max_image_size_mb=None).max_image_size_mb is None


def test_multimodal_globals_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("FASTAIAGENT_PDF_MODE", "vision")
    monkeypatch.setenv("FASTAIAGENT_MAX_PDF_PAGES", "4")
    monkeypatch.setenv("FASTAIAGENT_MAX_IMAGE_SIZE_MB", "2.5")
    reset_config()
    assert fa.config.pdf_mode == "vision"
    assert fa.config.max_pdf_pages == 4
    assert fa.config.max_image_size_mb == 2.5
    assert _client().max_pdf_pages == 4


def test_llm_client_dict_round_trip_survives_the_sentinel_defaults():
    fa.config.pdf_mode = "vision"
    fa.config.max_pdf_pages = 6
    fa.config.max_image_size_mb = 7.5

    from fastaiagent.llm.client import LLMClient

    original = _client(pdf_mode="text", max_pdf_pages=2, max_image_size_mb=None)
    payload = original.to_dict()
    restored = LLMClient.from_dict(payload)
    assert restored.pdf_mode == "text"
    assert restored.max_pdf_pages == 2
    assert restored.max_image_size_mb is None

    # And one built from the globals round-trips to the same resolved values ...
    from_global = _client()
    restored_global = LLMClient.from_dict(from_global.to_dict())
    assert restored_global.pdf_mode == "vision"
    assert restored_global.max_pdf_pages == 6
    assert restored_global.max_image_size_mb == 7.5

    # ... without widening the serialized payload, which reaches the plane as
    # ``agent.llm.config`` and is what Replay reconstructs from.
    assert "pdf_mode" not in from_global.to_dict()
    assert "max_pdf_pages" not in from_global.to_dict()
    assert "max_image_size_mb" not in from_global.to_dict()


# ── consistency guard ──────────────────────────────────────────────────────


def _iter_text_files():
    sources = (
        (_REPO_ROOT / "docs", "**/*.md"),
        (_REPO_ROOT / "fastaiagent", "**/*.py"),
    )
    for base, pattern in sources:
        for path in base.glob(pattern):
            yield path, path.read_text(encoding="utf-8")


def test_every_referenced_config_attribute_exists():
    """Nothing may tell a user to set a field that isn't there.

    This one *is* a string check, deliberately: the thing being checked is a
    string — the text of a docs page and of an HTTP 404 body. It is paired with
    the behavioural round-trips above, which are what prove the fields do
    something.
    """
    probe = SDKConfig()
    pattern = re.compile(r"\b(?:fa|fastaiagent)\.config\.([A-Za-z_][A-Za-z0-9_]*)")
    bad: dict[str, set[str]] = {}
    for path, text in _iter_text_files():
        referenced = {m.group(1) for m in pattern.finditer(text)}
        # ``hasattr`` rather than ``model_fields`` so a documented *method*
        # (``fa.config.model_dump()``) is not reported as a missing field.
        missing = {name for name in referenced if not hasattr(probe, name)}
        if missing:
            bad[str(path.relative_to(_REPO_ROOT))] = missing
    assert not bad, f"fa.config.<attr> references that resolve to nothing: {bad}"


#: Documented ``FASTAIAGENT_*`` names the library deliberately does not read.
#: Each one is a decision, not an oversight.
_DOC_ONLY_ENV_VARS = {
    # Read by scripts/capture-regression-from-trace-screenshots.sh to relocate an
    # example's local store; the SDK itself has never honoured it.
    "FASTAIAGENT_HOME",
    # CI-only overrides consumed by tests/e2e, documented in docs/cli/index.md.
    "FASTAIAGENT_LIVE_OPENAI_MODEL",
    "FASTAIAGENT_LIVE_ANTHROPIC_MODEL",
    # Appears only inside user-authored deployment snippets (docker/modal/
    # replicate), where the *example* reads it and passes it to connect(target=).
    "FASTAIAGENT_PLATFORM_URL",
}


def test_every_referenced_env_var_is_read_somewhere():
    """``FASTAIAGENT_*`` named in the docs must be read by the library."""
    import ast

    read: set[str] = set()
    for path in (_REPO_ROOT / "fastaiagent").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        read |= set(re.findall(r'"(FASTAIAGENT_[A-Z0-9_]+)"', text))
        # typer's ``envvar=`` and any other string constant are covered by the
        # regex above; the AST parse just proves the file is real Python.
        ast.parse(text)

    documented: set[str] = set()
    for path in (_REPO_ROOT / "docs").rglob("*.md"):
        documented |= set(re.findall(r"FASTAIAGENT_[A-Z0-9_]+", path.read_text(encoding="utf-8")))

    undefined = sorted(documented - read - _DOC_ONLY_ENV_VARS)
    assert not undefined, f"documented but read by nothing in fastaiagent/: {undefined}"


# ── helpers ────────────────────────────────────────────────────────────────


def _mm_kwargs(client) -> dict:
    kwargs = client._provider_dict_kwargs()
    kwargs.pop("model", None)
    return {"model": client.model, **{k: v for k, v in kwargs.items() if k != "model"}}


def _tiny_pdf() -> bytes:
    """A minimal one-page PDF with a text object, built without a dependency."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length 44 >>\nstream\nBT /F1 12 Tf 20 100 Td (hello pdf) Tj ET\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)
