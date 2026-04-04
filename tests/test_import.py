"""Smoke test: verify the package imports correctly."""


def test_import_fastaiagent():
    import fastaiagent

    assert hasattr(fastaiagent, "__version__")


def test_version_format():
    from fastaiagent._version import __version__

    # Version should be a valid PEP 440 string
    assert __version__  # non-empty
    assert "0.1." in __version__
