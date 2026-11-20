#!/usr/bin/env python3
"""Check that the fastaiagent names a Python file uses exist in the installed SDK.

    python check_imports.py path/to/file.py [more files or folders ...]

Resolves every ``import fastaiagent...`` / ``from fastaiagent... import X`` and
every ``fa.X`` attribute access on a module alias against the *installed*
package, so a guessed or removed name fails here instead of at run time.
It runs no code from the files it checks.

Exit code: 0 clean, 1 at least one problem, 2 the SDK is not importable or a
path is unreadable. Warnings (``warn:``) never change the exit code unless
``--strict`` is given.

Needs only the standard library plus the installed ``fastaiagent``.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
import types
from pathlib import Path

ROOT = "fastaiagent"

# Names/modules that older answers still use. Matched on the dotted name an import
# or attribute chain resolves to (or fails to resolve to).
REMOVED: dict[str, str] = {
    "fastaiagent.integrations.langchain": (
        "removed in 1.96.0. Use enable_otel_capture() with LangChain's own OpenTelemetry"
        " instrumentation."
    ),
    "fastaiagent.integrations.crewai": (
        "removed in 1.96.0. Use enable_otel_capture() with CrewAI's own OpenTelemetry"
        " instrumentation."
    ),
    "fastaiagent.integrations.pydanticai": (
        "removed in 1.96.0. Use enable_otel_capture() with Pydantic AI's own OpenTelemetry"
        " instrumentation."
    ),
    "fastaiagent.integrations.agent_name": "removed in 1.96.0 with the framework adapters.",
    "fastaiagent.discover_tools": "removed. Use MCPTool.from_server(...) (needs the [mcp] extra).",
    "fastaiagent.tool.discover_tools": (
        "removed. Use MCPTool.from_server(...) (needs the [mcp] extra)."
    ),
    "fastaiagent.tool.mcp.discover_tools": (
        "removed. Use MCPTool.from_server(...) (needs the [mcp] extra)."
    ),
}

DEAD_ADDRESS = "app.fastaiagent.net"


class Report:
    def __init__(self) -> None:
        self.problems = 0
        self.warnings = 0
        self._lines: list[tuple[str, int, str]] = []

    def problem(self, path: Path, line: int, msg: str) -> None:
        self.problems += 1
        self._lines.append((str(path), line, f"error: {msg}"))

    def warn(self, path: Path, line: int, msg: str) -> None:
        self.warnings += 1
        self._lines.append((str(path), line, f"warn: {msg}"))

    def flush(self) -> None:
        for path, line, text in sorted(self._lines):
            print(f"{path}:{line}: {text}")


def _import(dotted: str) -> types.ModuleType | None:
    try:
        return importlib.import_module(dotted)
    except ImportError:
        return None


def _member(owner: types.ModuleType, name: str) -> tuple[bool, object | None]:
    """``owner.name`` the way Python finds it: an attribute first, else a submodule.

    Order matters here: ``fastaiagent.tool`` is both a submodule and (re-exported on the
    package) the ``@tool`` decorator, and ``fa.tool`` / ``from fastaiagent import tool`` give
    the decorator. Imports of a *module path* (``from fastaiagent.tool import X``) go through
    ``_import`` instead, which gives the submodule.
    """
    if hasattr(owner, name):
        return True, getattr(owner, name)
    sub = _import(f"{owner.__name__}.{name}")
    return (sub is not None), sub


def _hint(dotted: str) -> str:
    for prefix, text in REMOVED.items():
        if dotted == prefix or dotted.startswith(prefix + "."):
            return f" ({text})"
    return ""


def _warn_private(
    mod: str, name: str, imported: object, path: Path, line: int, rep: Report
) -> None:
    """A name imported from a private module (``fastaiagent._internal...``) that the
    package also exports: say to use the public import, which survives refactors."""
    if not any(part.startswith("_") for part in mod.split(".")):
        return
    root = _import(ROOT)
    if root is not None and getattr(root, name, None) is imported:
        rep.warn(path, line, f"import {name} from {ROOT} ({ROOT}.{name}), not from {mod}.")


def _check_file(path: Path, rep: Report) -> None:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        print(f"{path}: cannot read: {err}", file=sys.stderr)
        raise SystemExit(2) from err
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as err:
        rep.problem(path, err.lineno or 1, f"syntax error: {err.msg}")
        return

    for lineno, text in enumerate(source.splitlines(), 1):
        if DEAD_ADDRESS in text:
            rep.warn(
                path,
                lineno,
                f"{DEAD_ADDRESS} is not a live address. Never invent a platform address;"
                " take it from FASTAIAGENT_TARGET or the user.",
            )

    aliases: dict[str, types.ModuleType] = {}  # local name -> the module it is bound to

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name != ROOT and not item.name.startswith(ROOT + "."):
                    continue
                module = _import(item.name)
                if module is None:
                    rep.problem(path, node.lineno, f"cannot import {item.name}{_hint(item.name)}")
                    continue
                if item.asname:
                    aliases[item.asname] = module
                else:  # `import fastaiagent.a.b` binds the top package
                    root = _import(ROOT)
                    if root is not None:
                        aliases[ROOT] = root
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or (mod != ROOT and not mod.startswith(ROOT + ".")):
                continue
            owner = _import(mod)
            if owner is None:
                rep.problem(path, node.lineno, f"cannot import {mod}{_hint(mod)}")
                continue
            for item in node.names:
                if item.name == "*":
                    continue
                ok, imported = _member(owner, item.name)
                if not ok:
                    dotted = f"{mod}.{item.name}"
                    rep.problem(
                        path, node.lineno, f"cannot import {item.name} from {mod}{_hint(dotted)}"
                    )
                elif isinstance(imported, types.ModuleType):
                    aliases[item.asname or item.name] = imported
                else:
                    _warn_private(mod, item.name, imported, path, node.lineno, rep)

    reported: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain: list[str] = []
            cur: ast.expr = node
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if not isinstance(cur, ast.Name) or cur.id not in aliases:
                continue
            chain.reverse()
            obj: object | None = aliases[cur.id]
            for part in chain:
                if not isinstance(obj, types.ModuleType):
                    break  # past a class/function: attributes are not checked
                walked = f"{obj.__name__}.{part}"
                ok, nxt = _member(obj, part)
                if not ok:
                    if (node.lineno, walked) not in reported:
                        reported.add((node.lineno, walked))
                        rep.problem(path, node.lineno, f"{walked} does not exist{_hint(walked)}")
                    break
                obj = nxt

    _warn_connect(tree, path, rep, aliases)


def _warn_connect(
    tree: ast.AST, path: Path, rep: Report, aliases: dict[str, types.ModuleType]
) -> None:
    """``connect()`` has no default target since 1.100.0."""
    bare: set[str] = set()  # local names bound by `from fastaiagent... import connect [as x]`
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level:
            if node.module == ROOT or (node.module or "").startswith(ROOT + "."):
                bare.update(i.asname or i.name for i in node.names if i.name == "connect")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute):
            base = fn.value
            if fn.attr != "connect" or not (isinstance(base, ast.Name) and base.id in aliases):
                continue
        elif not (isinstance(fn, ast.Name) and fn.id in bare):
            continue
        has_target = any(k.arg == "target" for k in node.keywords) or any(
            k.arg is None for k in node.keywords
        )
        if not has_target and len(node.args) < 2:
            rep.warn(
                path,
                node.lineno,
                "connect() needs target=... (or FASTAIAGENT_TARGET in the environment);"
                " there is no default since 1.100.0.",
            )


def _collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(x for x in p.rglob("*.py") if ".venv" not in x.parts))
        elif p.is_file():
            files.append(p)
        else:
            print(f"{raw}: no such file or folder", file=sys.stderr)
            raise SystemExit(2)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("paths", nargs="+", help="Python files or folders to check")
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    args = ap.parse_args(argv)

    try:
        importlib.import_module(ROOT)
    except Exception as err:  # not only ImportError: say what actually failed
        print(
            f"fastaiagent could not be imported by {sys.executable}:"
            f" {type(err).__name__}: {err}\n"
            "If it is not installed, install it (pip install fastaiagent) in the environment"
            " you run this with. If it is installed, the error above is the reason; run"
            f' `{sys.executable} -c "import fastaiagent"` to see the full traceback.',
            file=sys.stderr,
        )
        return 2

    rep = Report()
    files = _collect(args.paths)
    for f in files:
        _check_file(f, rep)
    rep.flush()
    print(f"checked {len(files)} file(s): {rep.problems} error(s), {rep.warnings} warning(s)")
    if rep.problems or (args.strict and rep.warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
