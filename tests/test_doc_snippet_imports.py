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
import hashlib
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fence markers we honour. CommonMark also allows longer runs of the same
# character; ``startswith`` below accepts those as the opener and as the
# closer, which is all these docs use.
_FENCES = ("```", "~~~")

# Info strings that mean "this block is Python".
_PY_LANGS = frozenset({"python", "py"})


@dataclass(frozen=True)
class CodeBlock:
    """One fenced block from a Markdown file, with its language tag."""

    file: Path
    line: int  # 1-based line of the block's FIRST BODY line
    lang: str
    body: str  # dedented; ready for ``ast.parse``

    @property
    def relative(self) -> Path:
        return self.file.relative_to(REPO_ROOT)

    @property
    def digest(self) -> str:
        """Content address, stable when unrelated lines above it move."""
        return hashlib.sha1(self.body.encode("utf-8")).hexdigest()[:12]


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


def _extract_python_blocks(file: Path, text: str) -> list[CodeBlock]:
    """Return every ```python fenced block in ``text``, dedented.

    Walks lines and tracks fence state rather than pattern-matching pairs.
    The regex this replaced made an opening fence's language tag optional,
    so a *closing* ``` matched it too: fences paired up shifted by one and
    the prose between two blocks was handed back as Python. That both
    manufactured phantom blocks and hid real ones — 334 import sites were
    collected where a correct walk finds 479, and two genuinely broken
    imports (one in the docs homepage Quick Start) sat in the gap.

    Bodies are dedented because a block indented inside a list item is
    valid Python that ``ast.parse`` rejects for its leading whitespace.
    """
    blocks: list[CodeBlock] = []
    marker = ""
    info = ""
    body: list[str] = []
    start = 0
    inside = False

    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.lstrip()
        if not inside:
            if stripped.startswith(_FENCES):
                marker = stripped[:3]
                info = stripped[3:].strip()
                body, start, inside = [], lineno + 1, True
            continue
        # A closing fence repeats the marker and carries no info string.
        if stripped.startswith(marker) and not stripped[3:].strip():
            inside = False
            lang = info.split()[0].lower() if info else ""
            if lang in _PY_LANGS:
                blocks.append(
                    CodeBlock(
                        file=file,
                        line=start,
                        lang=lang,
                        body=textwrap.dedent("\n".join(body)),
                    )
                )
            continue
        body.append(raw)

    return blocks


def _extract_fastaiagent_imports(blocks: list[CodeBlock], file: Path) -> list[ImportSite]:
    """Pull every ``import fastaiagent...`` / ``from fastaiagent... import ...`` site."""
    sites: list[ImportSite] = []
    for block in blocks:
        block_line, body = block.line, block.body
        try:
            tree = ast.parse(body)
        except SyntaxError:
            # An unparseable block cannot yield import sites. It is not
            # waved through: ``test_documented_python_block_parses`` below
            # fails on any block that is not in the signed-off allowlist.
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


def _all_doc_blocks() -> list[CodeBlock]:
    blocks: list[CodeBlock] = []
    for f in _doc_files():
        blocks.extend(_extract_python_blocks(f, f.read_text(encoding="utf-8")))
    return blocks


def _all_doc_imports() -> list[ImportSite]:
    sites: list[ImportSite] = []
    for f in _doc_files():
        text = f.read_text(encoding="utf-8")
        for site in _extract_fastaiagent_imports(_extract_python_blocks(f, text), f):
            sites.append(site)
    return sites


@pytest.mark.parametrize("site", _all_doc_imports(), ids=lambda s: f"{s.file.name}:{s.line}")
def test_documented_fastaiagent_import_resolves(site: ImportSite) -> None:
    """Each documented ``fastaiagent`` import must resolve in a fresh namespace.

    A ``ModuleNotFoundError`` for a third-party module (``fastapi``,
    ``langchain``, ``qdrant_client``, …) means the documented snippet
    targets an *optional* extra that isn't installed in this test
    environment — skip rather than fail, so the bare-install matrix
    doesn't go red just because the snippet uses ``[ui]`` or ``[kb]``
    features. A failure inside ``fastaiagent.*`` always counts as a
    real bug (re-export drift, renamed symbol, etc.).
    """
    namespace: dict[str, object] = {}
    try:
        exec(compile(site.code, str(site.file), "exec"), namespace)
    except ModuleNotFoundError as e:
        missing = (e.name or "").split(".")[0]
        if missing and missing != "fastaiagent":
            pytest.skip(
                f"snippet at {site.file.name}:{site.line} needs optional "
                f"third-party module {missing!r} "
                f"(install with the matching ``fastaiagent[<extra>]``)"
            )
        relative = site.file.relative_to(REPO_ROOT)
        pytest.fail(
            f"Broken documented import at {relative}:{site.line}\n"
            f"    {site.code}\n"
            f"  → {type(e).__name__}: {e}\n"
            f"This statement appears in published docs / the README — "
            f"new users copy-paste it verbatim. Re-export the missing "
            f"symbol or fix the import path."
        )
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


# ---------------------------------------------------------------------------
# The class-closer — every documented Python block must parse
# ---------------------------------------------------------------------------
#
# Why this exists as a separate test rather than a stricter
# ``_extract_fastaiagent_imports``: that function *skips* a block it cannot
# parse, so an unparseable snippet reads as "nothing to check" instead of
# "something is wrong". That is the same shape as the bug class CLAUDE.md §2.4
# names — a check that cannot run must say so, never return a clean verdict.
# Thirty blocks had accumulated behind that skip.
#
# Entries are keyed on a hash of the block's *content*, not its line number,
# so an unrelated edit higher up the file does not silently re-point an
# allowlist entry at a different block. Edit an allowlisted snippet and its
# key changes, which forces the exemption to be signed off again.
#
# Each value is the reason the block is exempt. Adding one is a decision.

_UNPARSEABLE_ALLOWLIST: dict[str, str] = {
    # API signature listings: annotated parameters inside a *call* expression,
    # a ``def`` with no body, or a trailing ``-> ReturnType``. Correct as
    # reference documentation, not executable Python. Re-fencing them as
    # ```text would render them un-highlighted, which is a worse trade for
    # nine reference blocks than an explicit exemption here.
    "docs/tools/mcp-server.md:5c431d3d455c": "as_mcp_server() signature listing",
    "docs/platform/index.md:e67d6d737b08": "platform API surface listing",
    "docs/agents/dynamic-instructions.md:51fb9f07f4e1": "callable type signature",
    "docs/agents/teams.md:a1d6642e6b5a": "Supervisor() signature listing",
    "docs/agents/teams.md:6a1a1fa678ab": "Worker() signature listing",
    "docs/testing/index.md:8596e8cc4d60": "TestModel() signature listing",
    "docs/internals/evaluation-system.md:1083e51a6e19": "evaluate() signature listing",
    "docs/internals/evaluation-system.md:312670e0651c": "LLMJudge() signature listing",
    "docs/evaluation/pytest.md:444d78c8cd49": "evaluate_one() signature listing",
}


def _block_key(block: CodeBlock) -> str:
    return f"{block.relative.as_posix()}:{block.digest}"


@pytest.mark.parametrize(
    "block",
    _all_doc_blocks(),
    ids=lambda b: f"{b.file.name}:{b.line}",
)
def test_documented_python_block_parses(block: CodeBlock) -> None:
    """Every ```python block in the README / docs/ must be valid Python.

    A block that does not parse is a block no other guard can inspect: the
    import checker walks past it, and ``mkdocs build`` renders it without
    ever compiling it. Readers copy it anyway.
    """
    key = _block_key(block)
    try:
        ast.parse(block.body)
    except SyntaxError as e:
        if key in _UNPARSEABLE_ALLOWLIST:
            pytest.xfail(f"signed off: {_UNPARSEABLE_ALLOWLIST[key]}")
        pytest.fail(
            f"Documented Python block does not parse at "
            f"{block.relative}:{block.line}\n"
            f"  → {type(e).__name__}: {e.msg} (block line {e.lineno})\n"
            f"Readers copy this verbatim. Fix the snippet, or — if it is a "
            f"signature listing rather than runnable code — add\n"
            f'    "{key}": "<why>",\n'
            f"to _UNPARSEABLE_ALLOWLIST in {Path(__file__).name}."
        )
    else:
        assert key not in _UNPARSEABLE_ALLOWLIST, (
            f"{block.relative}:{block.line} now parses, but is still listed in "
            f"_UNPARSEABLE_ALLOWLIST under key {key!r}. Remove the entry — a "
            f"stale exemption hides the next regression in this file."
        )


def test_unparseable_allowlist_has_no_dead_entries() -> None:
    """Every allowlist key must still match a block that exists.

    Without this, deleting or rewriting a snippet leaves an entry behind that
    exempts nothing and reads as though it still does.
    """
    live = {_block_key(b) for b in _all_doc_blocks()}
    dead = sorted(set(_UNPARSEABLE_ALLOWLIST) - live)
    assert not dead, (
        "_UNPARSEABLE_ALLOWLIST entries match no block in the docs — the "
        f"snippet was edited or removed, so drop them: {dead}"
    )
