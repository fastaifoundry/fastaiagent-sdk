"""Headless Agent Studio: ``fa.studio`` (plane 0.20.0+, wire v1.12).

    from fastaiagent.studio import StudioClient, StudioError

A coding tool designs the agent with its own model and sends the design as a spec;
Studio checks it, renders the code, runs trials and evals, keeps versions and exports.
The ``fastaiagent studio`` commands and the ``fastaiagent-studio`` skill sit on top of
this client.
"""

from fastaiagent._internal.errors import StudioError
from fastaiagent.studio.client import StudioClient
from fastaiagent.studio.models import (
    Draft,
    DraftDetail,
    EvalRollup,
    Fixture,
    Sandbox,
    SpecCheck,
    SpecProblem,
    SpecSchema,
    SpecWarning,
    StudioContext,
    Trial,
    Version,
    VersionAccepted,
    VersionDiff,
    VersionSummary,
)

__all__ = [
    "StudioClient",
    "StudioError",
    "StudioContext",
    "SpecSchema",
    "Draft",
    "DraftDetail",
    "Fixture",
    "SpecCheck",
    "SpecProblem",
    "SpecWarning",
    "VersionAccepted",
    "VersionSummary",
    "Version",
    "VersionDiff",
    "Sandbox",
    "Trial",
    "EvalRollup",
]
