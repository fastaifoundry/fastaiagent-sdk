"""Smoke test: verify the package imports correctly."""


def test_import_fastaiagent():
    import fastaiagent

    assert hasattr(fastaiagent, "__version__")


def test_version_format():
    from fastaiagent._version import __version__

    assert __version__ == "0.1.0a1"
