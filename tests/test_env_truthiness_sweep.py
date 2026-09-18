"""The class-closer for environment-switch truthiness.

In the spirit of ``tests/test_guardrail_unusable_config_sweep.py``: the point is
not to test one variable, it is to make the *class* of defect impossible to ship
again. Three consecutive audits found the same shape — a documented boolean that
only honoured a subset of its documented spellings, so ``FASTAIAGENT_TRACE_PAYLOADS=false``
kept egressing payloads and ``FASTAIAGENT_EXPORT_EVALS=yes`` silently disabled
export.

Two arms:

* **Behavioural.** Every entry of :data:`fastaiagent._internal.env.ENV_FLAGS` is
  driven over the full true-set and false-set (mixed case, surrounding
  whitespace) through **its real resolver at the real call site**, never through
  ``env_flag`` alone — CLAUDE.md §3: a test that exercises the helper it is
  supposed to be checking certifies rather than checks.
* **Discovery.** An AST walk of ``fastaiagent/`` plus the variables named in
  ``docs/configuration/environment-variables.md``, failing when a
  boolean-shaped variable has no registry entry.
"""

from __future__ import annotations

import ast
import importlib
import logging
import re
from pathlib import Path

import pytest

from fastaiagent._internal.config import reset_config
from fastaiagent._internal.env import ENV_FLAGS, FALSE_VALUES, TRUE_VALUES, EnvFlagSpec

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "fastaiagent"
_ENV_DOC = _REPO_ROOT / "docs" / "configuration" / "environment-variables.md"


def _resolve(spec: EnvFlagSpec):
    module_name, func_name = spec.resolver.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _call(name: str, spec: EnvFlagSpec) -> bool:
    """Drive the real resolver.

    ``reset_config()`` first because three of the resolvers read the
    ``lru_cache``d :class:`SDKConfig`, which in a real process is built once at
    the value the environment held at import time.
    """
    reset_config()
    return _resolve(spec)()


def _spellings(values: frozenset[str]) -> list[str]:
    """Each spelling, plus a mixed-case and a whitespace-padded variant."""
    out: list[str] = []
    for v in sorted(values):
        out.extend([v, v.upper(), v.capitalize(), f"  {v} ", f"\t{v}\n"])
    return out


_FLAG_NAMES = sorted(ENV_FLAGS)


@pytest.mark.parametrize("name", _FLAG_NAMES)
@pytest.mark.parametrize("value", _spellings(TRUE_VALUES))
def test_every_flag_reads_every_true_spelling(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    assert _call(name, ENV_FLAGS[name]) is True, (
        f"{name}={value!r} did not resolve to True through {ENV_FLAGS[name].resolver}"
    )


@pytest.mark.parametrize("name", _FLAG_NAMES)
@pytest.mark.parametrize("value", _spellings(FALSE_VALUES))
def test_every_flag_reads_every_false_spelling(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    assert _call(name, ENV_FLAGS[name]) is False, (
        f"{name}={value!r} did not resolve to False through {ENV_FLAGS[name].resolver}"
    )


@pytest.mark.parametrize("name", _FLAG_NAMES)
def test_unset_resolves_to_default(name, monkeypatch):
    monkeypatch.delenv(name, raising=False)
    assert _call(name, ENV_FLAGS[name]) is ENV_FLAGS[name].default


@pytest.mark.parametrize("name", _FLAG_NAMES)
@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_empty_resolves_to_default(name, value, monkeypatch):
    """A docker-compose ``FOO:`` or a k8s ``value: ""`` means "unset"."""
    monkeypatch.setenv(name, value)
    assert _call(name, ENV_FLAGS[name]) is ENV_FLAGS[name].default


@pytest.mark.parametrize("name", _FLAG_NAMES)
def test_unparsed_resolves_to_the_declared_side_and_warns(name, monkeypatch, caplog):
    spec = ENV_FLAGS[name]
    expected = spec.default if spec.on_unparsed is None else spec.on_unparsed
    monkeypatch.setenv(name, "ture")
    with caplog.at_level(logging.WARNING, logger="fastaiagent._internal.env"):
        assert _call(name, spec) is expected
    warnings = [
        r for r in caplog.records if r.levelno >= logging.WARNING and name in r.getMessage()
    ]
    assert warnings, f"no WARNING naming {name} was emitted for an unparseable value"
    assert "ture" in warnings[0].getMessage()


@pytest.mark.parametrize("name", sorted(n for n, s in ENV_FLAGS.items() if s.security_relevant))
def test_security_switches_fail_closed(name):
    """Signed-off 1.67.0 decision — a typo on a safety switch takes the safe side.

    For an opt-out (default on) that means off; for a capability grant (default
    off) it means not granted. Either way ``on_unparsed`` must be ``False``.
    """
    spec = ENV_FLAGS[name]
    assert spec.on_unparsed is False, (
        f"{name} is security-relevant but does not declare a fail-closed on_unparsed"
    )


# ── Discovery arm ──────────────────────────────────────────────────────────


#: Variables that look boolean to the scanners below but deliberately are not
#: registered. Every entry is a decision, not a formality — say why.
_NOT_BOOLEAN_ALLOWLIST: dict[str, str] = {
    # Tri-state: false-ish / true-ish / a CA-bundle path. Reuses TRUE_VALUES and
    # FALSE_VALUES but cannot be an env_flag, because the third state is a str.
    "FASTAIAGENT_LLM_VERIFY": "tri-state (bool or CA-bundle path)",
    # An enum, not a boolean: only the literal "closed" hardens the gate, which
    # is deliberate so a typo can never silently fail-closed a production gate.
    # It warns on an unrecognised value (see client._log/connect).
    "FASTAIAGENT_GOVERNANCE_FAIL_MODE": "enum: open | closed",
    # Free-form strings / paths / ints, never compared to a boolean literal.
    "FASTAIAGENT_API_KEY": "credential",
    "FASTAIAGENT_TARGET": "URL",
    "FASTAIAGENT_CONSOLE_URL": "URL",
    "FASTAIAGENT_PROJECT": "identifier",
    "FASTAIAGENT_LOCAL_DB": "path",
    "FASTAIAGENT_TRACE_DB_PATH": "path",
    "FASTAIAGENT_CHECKPOINT_DB_PATH": "path",
    "FASTAIAGENT_PROMPT_DIR": "path",
    "FASTAIAGENT_CACHE_DIR": "path",
    "FASTAIAGENT_KB_DIR": "path",
    "FASTAIAGENT_MODEL_CATALOG": "path",
    "FASTAIAGENT_UI_HOST": "host",
    "FASTAIAGENT_UI_PORT": "port",
    "FASTAIAGENT_UI_ALLOWED_HOSTS": "comma-separated host list",
    "FASTAIAGENT_SERVE_TOKEN": "bearer token",
    "FASTAIAGENT_LOG_LEVEL": "log level name",
    "FASTAIAGENT_DEFAULT_TIMEOUT": "seconds",
    "FASTAIAGENT_PDF_MODE": "enum: auto | text | vision | native",
    "FASTAIAGENT_MAX_IMAGE_SIZE_MB": "megabytes",
    "FASTAIAGENT_MAX_PDF_PAGES": "page count",
}

_BOOLISH_LITERALS = TRUE_VALUES | FALSE_VALUES


class _EnvBoolVisitor(ast.NodeVisitor):
    """Collect ``os.environ.get("X")`` / ``os.getenv("X")`` reads that are used
    as booleans — compared to a boolean-ish string literal, membership-tested
    against one, or passed to ``bool()``."""

    def __init__(self) -> None:
        self.found: set[str] = set()

    @staticmethod
    def _env_var_name(node: ast.AST) -> str | None:
        """Unwrap ``os.environ.get("X", "")[.strip()][.lower()]`` down to ``X``."""
        while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("strip", "lower", "upper", "casefold"):
                node = node.func.value
                continue
            break
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return None
        if node.func.attr not in ("get", "getenv"):
            return None
        if not node.args or not isinstance(node.args[0], ast.Constant):
            return None
        value = node.args[0].value
        return value if isinstance(value, str) and value.startswith("FASTAIAGENT_") else None

    def visit_Compare(self, node: ast.Compare) -> None:  # noqa: N802 — ast API
        name = self._env_var_name(node.left)
        if name is not None:
            for comparator in node.comparators:
                literals: list[str] = []
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    literals = [comparator.value]
                elif isinstance(comparator, (ast.Set, ast.Tuple, ast.List)):
                    literals = [
                        e.value
                        for e in comparator.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    ]
                if any(lit.lower() in _BOOLISH_LITERALS for lit in literals):
                    self.found.add(name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 — ast API
        if isinstance(node.func, ast.Name) and node.func.id == "bool" and node.args:
            name = self._env_var_name(node.args[0])
            if name is not None:
                self.found.add(name)
        self.generic_visit(node)


def _scan_source() -> set[str]:
    found: set[str] = set()
    for path in _PACKAGE_ROOT.rglob("*.py"):
        visitor = _EnvBoolVisitor()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        found |= visitor.found
    return found


def _scan_docs() -> set[str]:
    text = _ENV_DOC.read_text(encoding="utf-8")
    names = set(re.findall(r"FASTAIAGENT_[A-Z0-9_]+", text))
    boolish: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("`") for c in line.split("|")]
        row_names = [c for c in cells if c.startswith("FASTAIAGENT_")]
        if not row_names:
            continue
        default = cells[2].strip("`").lower() if len(cells) > 2 else ""
        if default in _BOOLISH_LITERALS:
            boolish |= set(row_names)
    # Anything documented at all still has to be classified, so the allowlist
    # stays an inventory rather than a dumping ground.
    return names | boolish


def test_every_boolean_shaped_variable_is_registered():
    candidates = _scan_source() | _scan_docs()
    unclassified = sorted(
        name for name in candidates if name not in ENV_FLAGS and name not in _NOT_BOOLEAN_ALLOWLIST
    )
    assert not unclassified, (
        "these FASTAIAGENT_* variables are neither in ENV_FLAGS nor in the "
        "not-a-boolean allowlist — classify each one deliberately: "
        f"{unclassified}"
    )


def test_no_raw_boolean_env_comparison_survives_outside_the_helper():
    """The helper is only worth having if nothing routes around it.

    Behavioural sibling of the sweep above, and the reason it is an AST walk
    rather than a grep: the old call sites compared ``os.environ.get(...)``
    against ``"0"`` / ``{"1", "true", ...}`` directly, which is exactly the
    pattern that drifted. ``_internal/env.py`` itself is the one place allowed
    to know those literals.
    """
    offenders: dict[str, set[str]] = {}
    for path in _PACKAGE_ROOT.rglob("*.py"):
        if path.name == "env.py" and path.parent.name == "_internal":
            continue
        visitor = _EnvBoolVisitor()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        if visitor.found:
            offenders[str(path.relative_to(_REPO_ROOT))] = visitor.found
    assert not offenders, (
        "boolean environment comparisons outside fastaiagent/_internal/env.py — "
        f"route them through env_flag(): {offenders}"
    )


def test_registry_documents_every_flag():
    """A registered flag must also appear in the canonical docs table."""
    text = _ENV_DOC.read_text(encoding="utf-8")
    missing = sorted(name for name in ENV_FLAGS if name not in text)
    assert not missing, f"registered but undocumented: {missing}"
