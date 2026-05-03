"""Project scoping — make ``.fastaiagent/`` the unit of isolation.

When the SDK writes any record (trace span, checkpoint, prompt, eval
run, guardrail event, attachment), it stamps it with a ``project_id``.
The UI reads only records for the current project. Same Postgres can
host multiple projects without cross-contamination.

The project_id comes from ``./.fastaiagent/config.toml`` — created
automatically on first execution (NOT on import; library imports stay
side-effect-free). Default project_id is the current directory's name.

This module is intentionally lazy:

* ``import fastaiagent`` does NOT touch the filesystem.
* The first call to :func:`get_project_id` triggers the
  load-or-create flow.
* Subsequent calls return the cached value.

CLI / tests can override via :func:`set_project_id` for isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from fastaiagent._version import __version__

CONFIG_DIR = ".fastaiagent"
CONFIG_FILE = "config.toml"
DEFAULT_GITIGNORE = "local.db\nlocal.db-wal\nlocal.db-shm\n"


@dataclass(frozen=True)
class ProjectConfig:
    """In-memory representation of ``.fastaiagent/config.toml``."""

    project_id: str
    created_at: str  # ISO date
    sdk_version: str

    def to_toml(self) -> str:
        return (
            f'project_id = "{self.project_id}"\n'
            f'created_at = "{self.created_at}"\n'
            f'sdk_version = "{self.sdk_version}"\n'
        )

    @classmethod
    def from_toml(cls, text: str) -> ProjectConfig:
        # Tiny TOML reader — we only emit three string keys, so keep the
        # dependency surface small. tomllib (3.11+) would also work.
        out: dict[str, str] = {}
        for line in text.splitlines():
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            out[key] = value
        return cls(
            project_id=out.get("project_id", "default"),
            created_at=out.get("created_at", date.today().isoformat()),
            sdk_version=out.get("sdk_version", __version__),
        )


_INSTANCE: ProjectConfig | None = None
_OVERRIDE: str | None = None


def _config_path(root: Path | None = None) -> Path:
    return (root or Path.cwd()) / CONFIG_DIR / CONFIG_FILE


def _gitignore_path(root: Path | None = None) -> Path:
    return (root or Path.cwd()) / CONFIG_DIR / ".gitignore"


def load_or_create() -> ProjectConfig:
    """Read ``.fastaiagent/config.toml`` or create it on first call.

    Side effects:
      * On first call, creates ``.fastaiagent/`` and the config + gitignore.
      * Subsequent calls re-read from disk so external edits propagate.
    """
    path = _config_path()
    if path.exists():
        return ProjectConfig.from_toml(path.read_text())
    config = ProjectConfig(
        project_id=Path.cwd().name or "default",
        created_at=date.today().isoformat(),
        sdk_version=__version__,
    )
    write_config(config)
    return config


def write_config(config: ProjectConfig, root: Path | None = None) -> None:
    """Persist a ProjectConfig to ``.fastaiagent/config.toml`` + write a
    project-default ``.gitignore`` next to it.
    """
    path = _config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.to_toml())
    gi = _gitignore_path(root)
    if not gi.exists():
        gi.write_text(DEFAULT_GITIGNORE)


def get_project_id() -> str:
    """Return the active project_id.

    Resolution order:
      1. Explicit override via :func:`set_project_id` (used by tests).
      2. Cached singleton.
      3. ``.fastaiagent/config.toml`` (loaded; creates the file on first
         call from a fresh directory).
    """
    if _OVERRIDE is not None:
        return _OVERRIDE
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = load_or_create()
    return _INSTANCE.project_id


def get_project_config() -> ProjectConfig:
    """Return the full ProjectConfig (project_id, created_at, sdk_version)."""
    if _INSTANCE is not None:
        return _INSTANCE
    return load_or_create()


def set_project_id(project_id: str | None) -> None:
    """Override the resolved project_id for the current process.

    Pass ``None`` to clear. Tests use this to scope writes without
    polluting the working directory's ``.fastaiagent/`` folder.
    """
    global _OVERRIDE
    _OVERRIDE = project_id


def reset_for_testing() -> None:
    """Clear the cached singleton + any override.

    Test fixtures call this between tests so each ``tmp_path`` works
    starts from scratch.
    """
    global _INSTANCE, _OVERRIDE
    _INSTANCE = None
    _OVERRIDE = None


def safe_get_project_id() -> str:
    """Like :func:`get_project_id` but never raises.

    Used at SDK write sites — the first call from a fresh directory
    creates ``.fastaiagent/config.toml`` (matching the spec's
    "first execution creates the project config" contract). If the
    filesystem is read-only or any other ``OSError`` happens, falls
    back to the cwd basename without raising; the DB column has a
    ``NOT NULL DEFAULT ''`` so the row still inserts.
    """
    global _INSTANCE
    try:
        if _OVERRIDE is not None:
            return _OVERRIDE
        if _INSTANCE is not None:
            return _INSTANCE.project_id
        # Resolve via load_or_create so the first SDK write from a
        # fresh project dir lands the config.toml + .gitignore.
        try:
            _INSTANCE = load_or_create()
            return _INSTANCE.project_id
        except OSError:
            # Read-only fs or similar — fall through to a non-persistent
            # best-effort id so the SDK still writes the row.
            return Path.cwd().name or ""
    except OSError:
        return ""


__all__ = [
    "ProjectConfig",
    "get_project_config",
    "get_project_id",
    "load_or_create",
    "reset_for_testing",
    "safe_get_project_id",
    "set_project_id",
    "write_config",
]
