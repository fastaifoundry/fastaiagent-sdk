#!/usr/bin/env python3
"""Build the contents of the public skills repository (fastaifoundry/fastaiagent-skills).

    python scripts/sync_skills_repo.py --target ../fastaiagent-skills          # write
    python scripts/sync_skills_repo.py --target ../fastaiagent-skills --check  # verify only

The skill folders under ``fastaiagent/skills/`` are the source of truth. This copies them
byte-for-byte to ``<target>/skills/<name>/`` and writes the repository's own files from
``scripts/skills_repo/`` (README, the Claude Code marketplace manifest) plus the
SDK's LICENSE. It never touches ``<target>/.git``.

``--check`` changes nothing and exits 1 when the target differs from what a sync would
write — a missing file, a changed byte, or a file that should not be there. Run it before
publishing a release.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILLS_SRC = REPO / "fastaiagent" / "skills"
TEMPLATES = Path(__file__).resolve().parent / "skills_repo"
IGNORED_DIRS = {"__pycache__", ".git"}
# Things that live in the target repo and are not generated.
KEEP = {".git", ".github"}


def sdk_version() -> str:
    ns: dict[str, str] = {}
    exec((REPO / "fastaiagent" / "_version.py").read_text(encoding="utf-8"), ns)  # noqa: S102
    return ns["__version__"]


def expected_tree() -> dict[str, bytes]:
    """Relative path (posix) -> exact bytes the target must contain."""
    files: dict[str, bytes] = {}
    for skill in sorted(p for p in SKILLS_SRC.iterdir() if (p / "SKILL.md").is_file()):
        for f in sorted(skill.rglob("*")):
            if f.is_file() and not (IGNORED_DIRS & set(f.relative_to(SKILLS_SRC).parts)):
                files[f"skills/{f.relative_to(SKILLS_SRC).as_posix()}"] = f.read_bytes()
    version = sdk_version()
    skills = sorted(p.name for p in SKILLS_SRC.iterdir() if (p / "SKILL.md").is_file())
    for t in sorted(p for p in TEMPLATES.rglob("*") if p.is_file()):
        text = t.read_text(encoding="utf-8")
        text = text.replace("{{sdk_version}}", version).replace(
            "{{skill_list}}", "\n".join(f"- `{name}`" for name in skills)
        )
        files[t.relative_to(TEMPLATES).as_posix()] = text.encode("utf-8")
    files["LICENSE"] = (REPO / "LICENSE").read_bytes()
    return files


def actual_tree(target: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    if not target.is_dir():
        return files
    for f in sorted(target.rglob("*")):
        rel = f.relative_to(target)
        if f.is_file() and not (KEEP & set(rel.parts)) and not (IGNORED_DIRS & set(rel.parts)):
            files[rel.as_posix()] = f.read_bytes()
    return files


def differences(target: Path) -> list[str]:
    want, have = expected_tree(), actual_tree(target)
    out = [f"missing: {p}" for p in sorted(want.keys() - have.keys())]
    out += [f"unexpected: {p}" for p in sorted(have.keys() - want.keys())]
    out += [
        f"differs: {p}" for p in sorted(k for k in want.keys() & have.keys() if want[k] != have[k])
    ]
    return out


def sync(target: Path) -> None:
    want = expected_tree()
    for rel in sorted(actual_tree(target).keys() - want.keys()):
        (target / rel).unlink()
    for rel, data in want.items():
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    # drop folders the deletions emptied (never .git / .github)
    for d in sorted((p for p in target.rglob("*") if p.is_dir()), reverse=True):
        rel = d.relative_to(target)
        if not (KEEP & set(rel.parts)) and not any(d.iterdir()):
            shutil.rmtree(d)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--target", required=True, type=Path, help="checkout of the skills repository")
    ap.add_argument("--check", action="store_true", help="verify only; exit 1 on any difference")
    args = ap.parse_args(argv)

    if args.check:
        diffs = differences(args.target)
        for d in diffs:
            print(d)
        print(f"{len(diffs)} difference(s)")
        return 1 if diffs else 0
    sync(args.target)
    print(f"synced {len(expected_tree())} files to {args.target} (SDK {sdk_version()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
