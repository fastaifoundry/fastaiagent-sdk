"""Resolve ``module:attr`` / ``path/to/file.py:attr`` specs into live objects.

Shared by ``fastaiagent optimize``, ``fastaiagent eval run``, ``fastaiagent
agent`` and ``fastaiagent mcp`` so every CLI accepts the same target syntax.
Raises plain :class:`ValueError` so it stays CLI-framework-agnostic; Typer
callers wrap it into ``BadParameter``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast


def resolve_target(spec: str) -> Any:
    """Resolve ``path/to/file.py:attr`` or ``pkg.module:attr`` into a live object."""
    if ":" not in spec:
        raise ValueError(f"Expected 'path/to/file.py:attr' or 'pkg.module:attr', got {spec!r}")
    module_part, attr = spec.rsplit(":", 1)
    path = Path(module_part)
    if path.exists():
        module_name = path.stem
        spec_obj = importlib.util.spec_from_file_location(module_name, str(path))
        if spec_obj is None or spec_obj.loader is None:
            raise ValueError(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.path.insert(0, str(path.parent.resolve()))
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)
    if not hasattr(module, attr):
        raise ValueError(f"Module {module_part!r} has no attribute {attr!r}")
    return getattr(module, attr)


def resolve_agent_fn(obj: Any) -> Callable[..., Any]:
    """Coerce a resolved target into a callable an eval loop can drive.

    Accepts a plain callable, or any object exposing ``.run`` (e.g. an
    ``Agent``); ``Agent.run`` also gives eval access to ``.output`` /
    ``.trace_id`` on its result.
    """
    run = getattr(obj, "run", None)
    if not callable(obj) and callable(run):
        return cast("Callable[..., Any]", run)
    if callable(obj):
        return cast("Callable[..., Any]", obj)
    raise ValueError(
        f"Target {obj!r} is neither callable nor an object with a callable .run method"
    )
