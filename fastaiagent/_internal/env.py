"""One truthiness parser for every environment switch the SDK reads.

Before this module each call site rolled its own comparison, and they disagreed:
``FASTAIAGENT_TRACE_PAYLOADS`` honoured only ``!= "0"`` (so ``false``/``no``/``off``
left payloads egressing), ``FASTAIAGENT_EXPORT_EVALS`` honoured only ``1``/``true``
(so ``yes``/``on`` *disabled* export), and the four newer security switches
accepted the full ``1/true/yes/on`` set. Meanwhile
``docs/configuration/environment-variables.md`` promised one behaviour for all of
them. This module makes that promise checkable — see
``tests/test_env_truthiness_sweep.py``, which parametrizes every entry of
:data:`ENV_FLAGS` over the full true- and false-sets *through the real call site*.

Three rules, and they are not negotiable per-variable:

1. **Unset and empty both mean "unset".** A docker-compose ``FOO:`` or a k8s
   ``value: ""`` renders an empty string, and the operator meant "I didn't set
   this", not "off". So an empty value resolves to the variable's *default*.
2. **Case and whitespace never matter.** ``" True "`` is ``True``.
3. **An unparseable value on a safety or egress switch fails closed.** Signed off
   for 1.67.0. ``FASTAIAGENT_TRACE_PAYLOADS=ture`` must not egress payloads just
   because the operator fat-fingered the opt-out; ``FASTAIAGENT_ALLOW_PRIVATE_NETWORKS=ture``
   must not grant the capability. Every such variable passes ``on_unparsed`` set to
   its restrictive side, and the parse *always* warns so the typo is visible.

Stdlib only — this sits under the clean-core rule (CLAUDE.md §3).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Spellings that mean "yes".
TRUE_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})
#: Spellings that mean "no".
FALSE_VALUES: frozenset[str] = frozenset({"0", "false", "no", "off"})

# Kept as private aliases because several call sites import the sets directly to
# do tri-state parsing (``FASTAIAGENT_LLM_VERIFY``) rather than a boolean.
_TRUE = TRUE_VALUES
_FALSE = FALSE_VALUES

_ACCEPTED = ", ".join(sorted(TRUE_VALUES) + sorted(FALSE_VALUES))


def env_flag(name: str, *, default: bool, on_unparsed: bool | None = None) -> bool:
    """Resolve ``name`` as a boolean environment switch.

    Arguments:
        name: the environment variable, e.g. ``"FASTAIAGENT_TRACE_PAYLOADS"``.
        default: what unset — or set-to-empty — resolves to.
        on_unparsed: what an unrecognised value resolves to. Omit it and an
            unrecognised value falls back to ``default``. **Security-relevant
            switches must pass this explicitly, set to the restrictive side**:
            for an opt-out that is "opted out", for a capability grant it is
            "not granted".

    An unrecognised value always logs a ``WARNING`` naming the variable, the
    offending value, and the accepted spellings — silence is how a typo becomes
    a leak.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if not value:
        # An empty value is "unset" — see the module docstring.
        return default
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    resolved = default if on_unparsed is None else on_unparsed
    logger.warning(
        "%s=%r is not a recognised boolean; using %s. Accepted values: %s.",
        name,
        raw,
        resolved,
        _ACCEPTED,
    )
    return resolved


def env_path(name: str, default: str | None = None) -> str | None:
    """Resolve ``name`` as a filesystem path, expanding ``~`` and ``$VARS``.

    Returns ``default`` when the variable is unset or empty. A bare ``str()``
    here is what made ``FASTAIAGENT_LOCAL_DB=~/x/local.db`` create a literal
    ``./~/`` directory under the current working directory — so two processes
    started from different directories got different stores and a resume found
    nothing.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    stripped = raw.strip()
    if not stripped:
        return default
    # ``normpath`` last, so a Windows operator writing the POSIX-looking
    # ``~/agents/local.db`` — which is what every doc and .env example shows —
    # gets ``C:\Users\me\agents\local.db`` rather than a mixed-separator
    # ``C:\Users\me/agents/local.db``. Both open the same file, but only one of
    # them compares equal to a path built any other way, and these values are
    # used as store identities.
    return os.path.normpath(os.path.expanduser(os.path.expandvars(stripped)))


@dataclass(frozen=True)
class EnvFlagSpec:
    """How one boolean environment variable resolves, and who reads it."""

    default: bool
    on_unparsed: bool | None
    security_relevant: bool
    #: Where the variable is actually consulted — dotted module path plus the
    #: callable, so the sweep test can drive the *real* resolver rather than
    #: re-testing :func:`env_flag` against itself.
    resolver: str
    summary: str


#: Every boolean environment variable the SDK reads.
#:
#: This registry is what turns ``docs/configuration/environment-variables.md``
#: from a promise into a contract: ``tests/test_env_truthiness_sweep.py``
#: enumerates it, drives each ``resolver`` over the full true/false sets, and
#: fails when a boolean-shaped variable appears in the source or the docs
#: without an entry here.
ENV_FLAGS: dict[str, EnvFlagSpec] = {
    "FASTAIAGENT_TRACE_ENABLED": EnvFlagSpec(
        default=True,
        on_unparsed=False,
        security_relevant=True,
        resolver="fastaiagent.trace.otel:tracing_enabled",
        summary="Master switch. Off means no local capture and no export.",
    ),
    "FASTAIAGENT_TRACE_PAYLOADS": EnvFlagSpec(
        default=True,
        on_unparsed=False,
        security_relevant=True,
        resolver="fastaiagent.trace.span:export_payloads_enabled",
        summary="Off strips payload attributes/events/status before egress.",
    ),
    "FASTAIAGENT_EXPORT_EVALS": EnvFlagSpec(
        default=True,
        on_unparsed=False,
        security_relevant=True,
        resolver="fastaiagent.eval.platform_export:_eval_export_env",
        summary="Off stops Agent-CI verdict metadata reaching the plane.",
    ),
    "FASTAIAGENT_EXPORT_CHECKPOINTS": EnvFlagSpec(
        default=True,
        on_unparsed=False,
        security_relevant=True,
        resolver="fastaiagent.client:_export_checkpoints_env",
        summary="Off stops checkpoint state replicating to the plane.",
    ),
    "FASTAIAGENT_RESTORE_FROM_PLANE": EnvFlagSpec(
        default=True,
        on_unparsed=False,
        security_relevant=True,
        resolver="fastaiagent.checkpointers.platform_replica:_restore_from_plane_enabled",
        summary="Off keeps a deliberately deleted run deleted (erasure control).",
    ),
    "FASTAIAGENT_DB_KEEP_PERMS": EnvFlagSpec(
        default=False,
        on_unparsed=False,
        security_relevant=True,
        resolver="fastaiagent._internal.storage:_keep_db_perms",
        summary="On leaves group/other access on local.db alone.",
    ),
    "FASTAIAGENT_UI_TRUST_PROXY": EnvFlagSpec(
        default=False,
        on_unparsed=False,
        security_relevant=True,
        resolver="fastaiagent.ui.throttle:_trust_proxy",
        summary="On makes the UI throttlers trust X-Forwarded-For.",
    ),
    "FASTAIAGENT_RUNNER_ALLOW_INSECURE": EnvFlagSpec(
        default=False,
        on_unparsed=False,
        security_relevant=True,
        resolver="fastaiagent.cli.runner:_allow_insecure_connect",
        summary="On allows a plaintext http runner --connect to a remote plane.",
    ),
    "FASTAIAGENT_ALLOW_PRIVATE_NETWORKS": EnvFlagSpec(
        default=False,
        on_unparsed=False,
        security_relevant=True,
        resolver="fastaiagent.multimodal._http:_allow_private_networks",
        summary="On lets the SSRF-guarded fetchers reach private hosts.",
    ),
    "FASTAIAGENT_UI_ENABLED": EnvFlagSpec(
        default=False,
        on_unparsed=None,
        security_relevant=False,
        resolver="fastaiagent._internal.config:_ui_enabled_env",
        summary="Enable the bundled local UI via config.",
    ),
    "FASTAIAGENT_TRACE_FULL_IMAGES": EnvFlagSpec(
        default=False,
        on_unparsed=None,
        security_relevant=False,
        resolver="fastaiagent._internal.config:_trace_full_images_env",
        summary="Store original image/PDF bytes next to the thumbnail in local.db.",
    ),
}


__all__ = [
    "ENV_FLAGS",
    "FALSE_VALUES",
    "TRUE_VALUES",
    "EnvFlagSpec",
    "env_flag",
    "env_path",
]
