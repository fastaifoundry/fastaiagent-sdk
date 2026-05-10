"""Regression: top-level ``import fastaiagent`` must work without pytest.

Before 1.10.1, ``fastaiagent/eval/__init__.py`` eagerly imported
``fastaiagent.eval.pytest_plugin``, which has ``import pytest`` at module
level. Any production install without pytest hit
``ModuleNotFoundError: No module named 'pytest'`` on the very first
``import fastaiagent`` line — broken developer experience for anyone
shipping the SDK to a non-test runtime.

The fix in ``eval/__init__.py`` wraps that one import in a try/except
and falls back to a stub that raises a clear ImportError only if the
``@case`` or ``@dataset`` decorator is actually called.

This test runs a fresh Python subprocess with a ``MetaPathFinder`` that
refuses to find ``pytest``, then confirms ``import fastaiagent``
succeeds.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_import_fastaiagent_succeeds_without_pytest() -> None:
    code = textwrap.dedent(
        """
        import sys

        class _BlockPytest:
            \"\"\"Refuse to resolve ``pytest`` so the import path falls
            through to the stub branch.\"\"\"

            def find_spec(self, name, path=None, target=None):
                if name == "pytest" or name.startswith("pytest."):
                    raise ImportError("pytest blocked by regression test")
                return None

        # Drop any cached pytest before installing the blocker.
        for mod in list(sys.modules):
            if mod == "pytest" or mod.startswith("pytest."):
                del sys.modules[mod]

        sys.meta_path.insert(0, _BlockPytest())

        import fastaiagent
        from fastaiagent.eval import case, pytest_dataset

        print("VERSION", fastaiagent.__version__)

        # The decorators must be callable; calling without pytest must
        # raise a clear ImportError pointing the user at the fix.
        for fn in (case, pytest_dataset):
            try:
                fn(input="x", expected="y")
            except ImportError as exc:
                assert "pytest" in str(exc).lower(), str(exc)
                print(f"STUB_OK {fn.__name__ if hasattr(fn, '__name__') else fn}")
            else:
                raise AssertionError("stub should have raised ImportError")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"import fastaiagent without pytest failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "VERSION" in result.stdout, result.stdout
    # Both stubs (case + pytest_dataset) must have raised ImportError.
    assert result.stdout.count("STUB_OK") == 2, result.stdout
