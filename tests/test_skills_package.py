"""The packaged Agent Skills stay true to the SDK they teach.

A skill is read by a coding assistant that was not trained on this SDK, so a wrong
import, a stale signature or a name that does not match its folder fails silently:
the assistant writes code against an API that does not exist, or the tool drops the
skill without saying so. Everything here checks behaviour:

* the Agent Skills format limits (name == folder, lengths, body size, one level deep);
* every fenced Python snippet parses and every ``fastaiagent`` name in it resolves,
  checked by the skill's own ``check_imports.py``;
* every ``signature`` fence matches ``inspect.signature`` of the real object;
* the skills are in the built wheel.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_ROOT = REPO_ROOT / "fastaiagent" / "skills"

_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_FENCE_RE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.S | re.M)
_LINK_RE = re.compile(r"(?:references|scripts)/[\w./-]+")


def _skill_dirs() -> list[Path]:
    return sorted(p for p in SKILLS_ROOT.iterdir() if (p / "SKILL.md").is_file())


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    assert text.startswith("---\n"), "SKILL.md must start with frontmatter"
    head, _, body = text[4:].partition("\n---\n")
    meta: dict[str, str] = {}
    for line in head.splitlines():
        key, sep, value = line.partition(":")
        assert sep, f"frontmatter line is not 'key: value': {line!r}"
        meta[key.strip()] = value.strip()
    return meta, body


def _fences(path: Path) -> list[tuple[str, int, str]]:
    """(language, first body line number, body) for every fence in a markdown file."""
    text = path.read_text(encoding="utf-8")
    out = []
    for m in _FENCE_RE.finditer(text):
        line = text.count("\n", 0, m.start(2)) + 1
        out.append((m.group(1), line, m.group(2)))
    return out


def _all_markdown() -> list[Path]:
    return sorted(p for d in _skill_dirs() for p in d.rglob("*.md"))


def _load_checker() -> ModuleType:
    path = SKILLS_ROOT / "fastaiagent-sdk" / "scripts" / "check_imports.py"
    spec = importlib.util.spec_from_file_location("skill_check_imports", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_there_are_skills() -> None:
    assert [d.name for d in _skill_dirs()], f"no skills under {SKILLS_ROOT}"


@pytest.mark.parametrize("skill", _skill_dirs(), ids=lambda p: p.name)
def test_frontmatter_follows_the_format(skill: Path) -> None:
    meta, body = _split_frontmatter((skill / "SKILL.md").read_text(encoding="utf-8"))
    # Other keys vary by tool; the open format needs only these two.
    assert set(meta) == {"name", "description"}, sorted(meta)
    # A name that differs from the folder makes some tools drop the skill silently.
    assert meta["name"] == skill.name
    assert len(meta["name"]) <= 64 and _NAME_RE.match(meta["name"]), meta["name"]
    assert 1 <= len(meta["description"]) <= 1024, len(meta["description"])
    assert len(body.splitlines()) < 500


@pytest.mark.parametrize("skill", _skill_dirs(), ids=lambda p: p.name)
def test_references_are_one_level_deep_and_linked(skill: Path) -> None:
    refs = skill / "references"
    if refs.exists():
        assert not [p for p in refs.iterdir() if p.is_dir()], "references/ must be flat"
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    for link in sorted(set(_LINK_RE.findall(text))):
        assert (skill / link.rstrip(".")).exists(), f"SKILL.md mentions {link}, which is missing"
    if refs.exists():
        for ref in sorted(refs.glob("*.md")):
            assert f"references/{ref.name}" in text, f"{ref.name} is not linked from SKILL.md"


@pytest.mark.parametrize("path", _all_markdown(), ids=lambda p: str(p.relative_to(SKILLS_ROOT)))
def test_markdown_python_snippets_parse_and_resolve(path: Path, tmp_path: Path) -> None:
    """Every python fence parses and every fastaiagent name in it exists in this build."""
    checker = _load_checker()
    rep = checker.Report()
    for lang, line, body in _fences(path):
        if lang not in ("python", "py"):
            continue
        try:
            ast.parse(body)
        except SyntaxError as err:
            pytest.fail(f"{path.name}:{line}: snippet does not parse: {err}")
        snippet = tmp_path / f"snippet_{line}.py"
        snippet.write_text(body, encoding="utf-8")
        checker._check_file(snippet, rep)
    assert rep.problems == 0, "\n".join(f"{p}:{n}: {t}" for p, n, t in sorted(rep._lines))


def _parse_signature_line(line: str) -> tuple[str, list[str]]:
    m = re.fullmatch(r"([\w.]+)\((.*)\)", line.strip())
    assert m, f"not 'dotted.path(params)': {line!r}"
    params = [p.strip() for p in m.group(2).split(",") if p.strip()]
    return m.group(1), params


def _walk(dotted: str, attribute_first: bool) -> object:
    parts = dotted.split(".")
    obj: object = importlib.import_module(parts[0])
    walked = parts[0]
    for part in parts[1:]:
        walked = f"{walked}.{part}"
        order = ("attr", "module") if attribute_first else ("module", "attr")
        for how in order:
            if how == "attr" and hasattr(obj, part):
                obj = getattr(obj, part)
                break
            if how == "module":
                try:
                    obj = importlib.import_module(walked)
                    break
                except ImportError:
                    continue
        else:
            raise AssertionError(f"cannot resolve {dotted}")
    return obj


def _resolve(dotted: str) -> object:
    """Resolve a dotted path. ``fastaiagent.eval.evaluate`` is a function that shares its name
    with a module, and ``fastaiagent.tool`` is the decorator *and* a package, so a path is
    accepted when either reading of it (attribute first, or module first) lands on an object."""
    try:
        return _walk(dotted, attribute_first=True)
    except AssertionError:
        return _walk(dotted, attribute_first=False)


def _signature_of(obj: object) -> inspect.Signature:
    target = obj.__init__ if inspect.isclass(obj) else obj  # type: ignore[misc]
    sig = inspect.signature(target)
    params = [p for name, p in sig.parameters.items() if name not in ("self", "cls")]
    return sig.replace(parameters=params)


@pytest.mark.parametrize("path", _all_markdown(), ids=lambda p: str(p.relative_to(SKILLS_ROOT)))
def test_quoted_signatures_match_the_real_ones(path: Path) -> None:
    """Names listed exist; every required parameter is listed."""
    problems: list[str] = []
    for lang, line, body in _fences(path):
        if lang != "signature":
            continue
        for offset, raw in enumerate(body.splitlines()):
            if not raw.strip():
                continue
            where = f"{path.name}:{line + offset}"
            try:
                dotted, listed = _parse_signature_line(raw)
                real = _signature_of(_resolve(dotted))
            except (AssertionError, AttributeError, TypeError, ValueError) as err:
                problems.append(f"{where}: {raw.strip()!r}: {err}")
                continue
            names = {p.name: p for p in real.parameters.values()}
            has_var_kw = any(p.kind is p.VAR_KEYWORD for p in names.values())
            listed_names = [p.lstrip("*").rstrip("=").strip() for p in listed]
            for n in listed_names:
                if n == "...":
                    continue
                if n not in names and not has_var_kw:
                    problems.append(f"{where}: {dotted} has no parameter {n!r}")
            for p in names.values():
                required = p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                if required and p.name not in listed_names:
                    problems.append(f"{where}: {dotted} requires {p.name!r}, not listed")
    assert not problems, "\n".join(problems)


def test_signature_check_rejects_a_wrong_name() -> None:
    """The check above can fail: a made-up parameter and a missing required one are caught."""
    real = _signature_of(_resolve("fastaiagent.connect"))
    assert "api_key" in real.parameters and "nonexistent" not in real.parameters
    assert _parse_signature_line("fastaiagent.connect(api_key, target=)") == (
        "fastaiagent.connect",
        ["api_key", "target="],
    )


def test_check_imports_script_flags_removed_and_invented_names(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from fastaiagent import Agent, NotARealName\n"
        "from fastaiagent.integrations.langchain import X\n"
        "from fastaiagent.tool import discover_tools\n"
        "import fastaiagent as fa\n"
        'fa.connect(api_key="k")\n'
        'URL = "https://app.fastaiagent.net"\n',
        encoding="utf-8",
    )
    script = SKILLS_ROOT / "fastaiagent-sdk" / "scripts" / "check_imports.py"
    proc = subprocess.run(
        [sys.executable, "-I", str(script), str(bad)], capture_output=True, text=True
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    out = proc.stdout
    assert "cannot import NotARealName" in out
    assert "removed in 1.96.0" in out
    assert "MCPTool.from_server" in out
    assert "connect() needs target" in out
    assert "not a live address" in out

    good = tmp_path / "good.py"
    good.write_text("from fastaiagent import Agent\nfrom fastaiagent.testing import TestModel\n")
    ok = subprocess.run(
        [sys.executable, "-I", str(script), str(good)], capture_output=True, text=True
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr


def test_check_imports_script_points_private_imports_at_the_public_name(tmp_path: Path) -> None:
    """An assistant copied ``from fastaiagent._internal.errors import GuardrailBlockedError``
    (once the only way) after it became public in 1.104.0. A name that is not exported
    publicly (``MaxIterationsError``) has no better import, so it is not flagged."""
    src = tmp_path / "private.py"
    src.write_text(
        "from fastaiagent._internal.errors import GuardrailBlockedError, MaxIterationsError\n",
        encoding="utf-8",
    )
    script = SKILLS_ROOT / "fastaiagent-sdk" / "scripts" / "check_imports.py"
    proc = subprocess.run(
        [sys.executable, "-I", str(script), str(src)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr  # a warning, not an error
    assert "import GuardrailBlockedError from fastaiagent" in proc.stdout
    assert "MaxIterationsError" not in proc.stdout
    assert "1 warning(s)" in proc.stdout


def test_skills_are_in_the_built_wheel(tmp_path: Path) -> None:
    """Hatch ships everything under fastaiagent/, but check it: a skill missing from the
    wheel is invisible to ``fastaiagent skills install`` and nothing else notices."""
    pytest.importorskip("hatchling", reason="hatchling is not installed in this environment")
    out = tmp_path / "wheel"
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out), str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"wheel build failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    (wheel,) = out.glob("*.whl")
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    for skill in _skill_dirs():
        for f in skill.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                rel = f.relative_to(REPO_ROOT).as_posix()
                assert rel in names, f"{rel} is not in {wheel.name}"


# --- the public skills repository is generated from the package ---------------------------


def _sync_module() -> ModuleType:
    path = REPO_ROOT / "scripts" / "sync_skills_repo.py"
    spec = importlib.util.spec_from_file_location("sync_skills_repo", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_skills_repo_copy_is_byte_identical_and_drift_is_caught(tmp_path: Path) -> None:
    sync = _sync_module()
    target = tmp_path / "fastaiagent-skills"
    (target / ".git").mkdir(parents=True)
    (target / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    assert sync.main(["--target", str(target)]) == 0
    assert sync.main(["--target", str(target), "--check"]) == 0
    for skill in _skill_dirs():
        for f in skill.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                copy = target / "skills" / f.relative_to(SKILLS_ROOT)
                assert copy.read_bytes() == f.read_bytes(), copy
    assert (target / ".git" / "HEAD").exists(), "the sync must not touch .git"

    # Drift in each direction fails the check.
    some = next((target / "skills").rglob("SKILL.md"))
    some.write_bytes(some.read_bytes() + b"\nedited in the public repo\n")
    assert sync.main(["--target", str(target), "--check"]) == 1
    sync.main(["--target", str(target)])
    assert sync.main(["--target", str(target), "--check"]) == 0

    (target / "skills" / "extra.md").write_text("not from the package")
    assert sync.main(["--target", str(target), "--check"]) == 1
    sync.main(["--target", str(target)])
    assert not (target / "skills" / "extra.md").exists()

    some.unlink()
    assert sync.main(["--target", str(target), "--check"]) == 1


def test_marketplace_manifest_is_valid_json_with_the_required_fields(tmp_path: Path) -> None:
    import json

    sync = _sync_module()
    sync.main(["--target", str(tmp_path)])
    manifest = json.loads((tmp_path / ".claude-plugin" / "marketplace.json").read_text())
    assert {"name", "owner", "plugins"} <= set(manifest)
    assert manifest["owner"]["name"]
    (plugin,) = manifest["plugins"]
    assert plugin["name"] and plugin["source"] == "./"
    # Claude Code finds the skills by scanning skills/<name>/SKILL.md under the plugin root.
    assert [p.name for p in sorted((tmp_path / "skills").iterdir())] == [
        d.name for d in _skill_dirs()
    ]
    readme = (tmp_path / "README.md").read_text()
    for d in _skill_dirs():
        assert f"`{d.name}`" in readme
    assert "{{" not in readme


def test_llms_txt_files_are_built_from_the_skill() -> None:
    """docs/llms.txt and llms-full.txt are generated; a stale copy would tell a tool one
    thing while the installed skill says another."""
    path = REPO_ROOT / "scripts" / "build_llms_txt.py"
    spec = importlib.util.spec_from_file_location("build_llms_txt", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stale = [
        str(p.relative_to(REPO_ROOT))
        for p, text in mod.outputs().items()
        if not p.exists() or p.read_text(encoding="utf-8") != text
    ]
    assert not stale, f"run: python scripts/build_llms_txt.py  (out of date: {stale})"
    full = (REPO_ROOT / "docs" / "llms-full.txt").read_text(encoding="utf-8")
    for ref in sorted((SKILLS_ROOT / "fastaiagent-sdk" / "references").glob("*.md")):
        assert ref.read_text(encoding="utf-8").strip() in full, ref.name
