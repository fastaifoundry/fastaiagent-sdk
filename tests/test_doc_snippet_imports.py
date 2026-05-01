"""Static check that every ``from fastaiagent... import X`` line in the
README and the published docs actually resolves at runtime.

Why: ``mkdocs build`` renders fenced Python blocks but does not execute
them, so a stale import path can sit in the README quickstart for
months without any test catching it. Real users copy README snippets
verbatim — a broken import there is a 0-day bug for every new user.

This test does NOT execute arbitrary snippet code (most snippets need
an LLM key or external state). It only re-runs the **import statements**
that reference ``fastaiagent``, in a fresh namespace, and asserts each
one resolves. Fast, deterministic, no network.

Caught: the 1.1.0 → 1.1.1 patch where ``from fastaiagent.trace import
Replay`` was documented in 4 places but the package init didn't
re-export ``Replay``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Match a fenced Python block. Allow ```python, ```py, or just ``` (treated
# as Python only when an import line is present, keeping false positives low).
_FENCE_RE = re.compile(
    r"^```(?P<lang>python|py)?\s*\n(?P<body>.*?)\n```",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class ImportSite:
    """A single ``import`` / ``from … import …`` statement extracted from a doc."""

    file: Path
    line: int
    code: str  # the verbatim import statement


def _doc_files() -> list[Path]:
    """Every Markdown file we publish — README plus everything under docs/.

    Skip the ``api-reference`` tree because it's auto-generated and may
    reference internal symbols that move around.
    """
    paths: list[Path] = [REPO_ROOT / "README.md"]
    for md in (REPO_ROOT / "docs").rglob("*.md"):
        if "api-reference" in md.parts:
            continue
        paths.append(md)
    return [p for p in paths if p.exists()]


def _extract_python_blocks(text: str) -> list[tuple[int, str]]:
    """Return (start_line, body) for every fenced Python block."""
    blocks: list[tuple[int, str]] = []
    for m in _FENCE_RE.finditer(text):
        body = m.group("body")
        # Cheap heuristic when the fence is bare ``` (no lang tag): only
        # treat it as Python when it contains an import line. Avoids
        # mistakenly running shell / JSON snippets.
        if m.group("lang") is None and "import " not in body:
            continue
        start_line = text.count("\n", 0, m.start()) + 2
        blocks.append((start_line, body))
    return blocks


def _extract_fastaiagent_imports(blocks: list[tuple[int, str]], file: Path) -> list[ImportSite]:
    """Pull every ``import fastaiagent...`` / ``from fastaiagent... import ...`` site."""
    sites: list[ImportSite] = []
    for block_line, body in blocks:
        try:
            tree = ast.parse(body)
        except SyntaxError:
            # Snippet uses ``...`` placeholder or pseudo-code — skip the
            # whole block rather than guess. The bug we're guarding
            # against is a real broken import in valid Python.
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if not module.startswith("fastaiagent"):
                    continue
                names = ", ".join(alias.name for alias in node.names)
                code = f"from {module} import {names}"
            elif isinstance(node, ast.Import):
                fa_aliases = [a for a in node.names if a.name.startswith("fastaiagent")]
                if not fa_aliases:
                    continue
                code = "import " + ", ".join(a.name for a in fa_aliases)
            else:
                continue
            absolute_line = block_line + (node.lineno - 1)
            sites.append(ImportSite(file=file, line=absolute_line, code=code))
    return sites


def _all_doc_imports() -> list[ImportSite]:
    sites: list[ImportSite] = []
    for f in _doc_files():
        text = f.read_text(encoding="utf-8")
        for site in _extract_fastaiagent_imports(_extract_python_blocks(text), f):
            sites.append(site)
    return sites


@pytest.mark.parametrize("site", _all_doc_imports(), ids=lambda s: f"{s.file.name}:{s.line}")
def test_documented_fastaiagent_import_resolves(site: ImportSite) -> None:
    """Each documented ``fastaiagent`` import must resolve in a fresh namespace."""
    namespace: dict[str, object] = {}
    try:
        exec(compile(site.code, str(site.file), "exec"), namespace)
    except ImportError as e:
        relative = site.file.relative_to(REPO_ROOT)
        pytest.fail(
            f"Broken documented import at {relative}:{site.line}\n"
            f"    {site.code}\n"
            f"  → {type(e).__name__}: {e}\n"
            f"This statement appears in published docs / the README — "
            f"new users copy-paste it verbatim. Re-export the missing "
            f"symbol or fix the import path."
        )
