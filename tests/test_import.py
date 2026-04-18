"""Smoke test: verify the package imports correctly."""


def test_import_fastaiagent():
    import fastaiagent

    assert hasattr(fastaiagent, "__version__")


def test_version_format():
    import re

    from fastaiagent._version import __version__

    # Version should be a non-empty PEP 440 string (MAJOR.MINOR.PATCH).
    assert __version__
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[ab]\d+|rc\d+)?", __version__), __version__
