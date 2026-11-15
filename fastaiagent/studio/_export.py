"""Unpack a Studio export (a zip of the project) into a folder, safely.

The plane puts the rendered files under one folder named after the draft. Unpacking
refuses three things a careless ``extractall`` would do: write outside the target
folder (a member named ``../x``), overwrite a file that is already there, and follow an
absolute path.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from fastaiagent._internal.errors import StudioError


def unpack_export(archive: str | Path, out_dir: str | Path, *, force: bool = False) -> list[Path]:
    """Write the zip's files under ``out_dir`` and return the paths written.

    Raises :class:`StudioError` (client-side codes ``unsafe_export`` and
    ``export_would_overwrite``) and writes **nothing** if any member is unsafe or any
    target file exists and ``force`` is false — it checks everything before it writes
    anything, so a refusal never leaves half a project behind.
    """
    root = Path(out_dir).resolve()
    with zipfile.ZipFile(archive) as bundle:
        members = [m for m in bundle.infolist() if not m.is_dir()]
        targets: list[tuple[zipfile.ZipInfo, Path]] = []
        for member in members:
            target = (root / member.filename).resolve()
            if root != target and root not in target.parents:
                raise StudioError(
                    "unsafe_export",
                    f"The export holds a path outside the target folder: {member.filename!r}. "
                    "Nothing was written.",
                )
            targets.append((member, target))
        clashes = [t for _, t in targets if t.exists()]
        if clashes and not force:
            shown = ", ".join(str(c.relative_to(root)) for c in clashes[:5])
            more = f" (and {len(clashes) - 5} more)" if len(clashes) > 5 else ""
            raise StudioError(
                "export_would_overwrite",
                f"{len(clashes)} file(s) already exist under {root}: {shown}{more}. "
                "Nothing was written; pass --force to overwrite, or choose another --out.",
            )
        written: list[Path] = []
        for member, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(member))
            written.append(target)
    return written
