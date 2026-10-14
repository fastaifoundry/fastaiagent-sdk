"""Prompt registry with versioning, aliases, fragment composition, and platform support."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from fastaiagent._internal.config import get_config
from fastaiagent._internal.errors import PromptNotFoundError
from fastaiagent.prompt.fragment import FRAGMENT_RE as _FRAGMENT_RE
from fastaiagent.prompt.fragment import Fragment, fragment_refs
from fastaiagent.prompt.prompt import Prompt
from fastaiagent.prompt.storage import SQLiteStorage

logger = logging.getLogger(__name__)

__all__ = ["PromptRegistry", "fragment_refs"]

_DEFAULT_CACHE_TTL = 300  # 5 minutes


class PromptRegistry:
    """Prompt registry backed by the unified local SQLite store.

    Example:
        reg = PromptRegistry()
        reg.register_fragment(name="tone", content="Be professional.")
        reg.register(name="greeting", template="Hello {{name}}. {{@tone}}")
        prompt = reg.load("greeting")
        text = prompt.format(name="World")

    With platform:
        fa.connect(api_key="fa-...", project="my-project")
        reg = PromptRegistry()
        prompt = reg.get("support-prompt")  # fetches from platform
    """

    def __init__(self, store: str = "local", path: str | None = None):
        resolved = path if path is not None else get_config().local_db_path
        self._storage = SQLiteStorage(resolved)
        self._fragments: dict[str, Fragment] = {}
        self._platform_cache: dict[tuple[str, int | None], tuple[Prompt, float]] = {}
        self._cache_ttl: int = _DEFAULT_CACHE_TTL

    def is_local(self) -> bool:
        """True iff the backing SQLite file resolves inside the current project tree.

        Gates the UI's prompt editor: when False the editor goes read-only and a
        banner explains the registry is owned by an external environment.
        """
        try:
            storage_file = self._storage.file.resolve()
            cwd = Path.cwd().resolve()
            storage_file.relative_to(cwd)
            return True
        except (ValueError, OSError):
            return False

    def get(
        self,
        slug: str,
        version: int | None = None,
        source: str = "auto",
    ) -> Prompt:
        """Get a prompt by slug.

        source: "auto" (platform if connected, else local),
                "platform" (platform only), "local" (local only)
        """
        if source == "platform" or (source == "auto" and self._is_connected()):
            prompt = self._fetch_from_platform(slug, version)
            if prompt:
                return prompt
            if source == "platform":
                from fastaiagent._internal.errors import PromptNotFoundError

                raise PromptNotFoundError(f"Prompt '{slug}' not found on platform")
        return self._fetch_from_local(slug, version)

    def publish(
        self,
        slug: str,
        content: str,
        variables: list[str] | None = None,
    ) -> None:
        """Publish a prompt to the platform registry."""
        from fastaiagent._internal.errors import PlatformNotConnectedError
        from fastaiagent._platform.api import get_platform_api

        if not self._is_connected():
            raise PlatformNotConnectedError(
                "Not connected to platform. Call fa.connect() first."
            )
        api = get_platform_api()
        api.post(
            "/public/v1/prompts",
            {
                "slug": slug,
                "content": content,
                "variables": variables or [],
            },
        )

    def refresh(self, slug: str) -> None:
        """Invalidate the platform cache for a prompt."""
        keys_to_remove = [k for k in self._platform_cache if k[0] == slug]
        for k in keys_to_remove:
            del self._platform_cache[k]

    def register(
        self,
        name: str,
        template: str,
        fragments: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        version: int | None = None,
        project_id: str | None = None,
        *,
        created_by: str = "code",
    ) -> Prompt:
        """Register a new prompt (creates a new version).

        ``created_by`` is recorded on the version row: ``"code"`` for calls
        like this one, ``"ui"`` for the Local UI's editor (1.89.0).

        ``project_id`` is forwarded to the storage layer so callers like
        the Local UI can stamp rows with the *active* project rather than
        the cwd-derived default. Defaults to the auto-detected project id
        for backwards compatibility.
        """
        try:
            existing = self._storage.load_prompt(name)
            new_version = existing.version + 1
        except Exception:
            new_version = 1

        if version is not None:
            new_version = version
            # A version is a frozen row. Before 1.88.0 an explicit ``version=``
            # that already existed was written with ``INSERT OR REPLACE``, so v1
            # silently became new text and every alias pinned to it served that
            # text. Re-registering the *same* template is a no-op (seed scripts
            # re-run); a different template under an existing number is refused.
            try:
                existing_version = self._storage.load_prompt(name, version=version)
            except PromptNotFoundError:
                existing_version = None
            if existing_version is not None:
                if existing_version.template == template and existing_version.metadata == (
                    metadata or {}
                ):
                    return existing_version
                raise ValueError(
                    f"Prompt {name!r} already has a version {version} with a different "
                    "template. Versions are immutable: omit version= to add the next one "
                    f"(v{self._storage.load_prompt(name).version + 1}), or delete the prompt "
                    "first if you really mean to rewrite its history."
                )

        prompt = Prompt(
            name=name,
            template=template,
            version=new_version,
            metadata=metadata or {},
        )
        self._storage.save_prompt(prompt, project_id=project_id, created_by=created_by)
        return prompt

    def delete(
        self, name: str, *, project_id: str | None = None
    ) -> int:
        """Delete a prompt (every version + every alias) from local storage.

        Returns the number of versions removed. ``0`` means the prompt
        didn't exist in the requested scope — the UI route turns that into
        a ``404`` so callers can distinguish "wasn't there" from "deleted
        successfully."
        """
        # Drop any cached platform copies so re-registering immediately
        # reads fresh data instead of a stale TTL hit.
        self.refresh(name)
        return self._storage.delete_prompt(name, project_id=project_id)

    def register_fragment(self, name: str, content: str) -> Fragment:
        """Register a reusable prompt fragment."""
        fragment = Fragment(name=name, content=content)
        self._storage.save_fragment(fragment)
        self._fragments[name] = fragment
        return fragment

    def load(
        self,
        name: str,
        version: int | None = None,
        alias: str | None = None,
        *,
        resolve_fragments: bool = True,
    ) -> Prompt:
        """Load a prompt from local storage, resolving {{@fragment}} references.

        ``resolve_fragments=False`` returns the template as stored, with its
        ``{{@fragment}}`` references intact. An editor must start from that
        text: saving the resolved text back replaced the reference with a copy,
        so later edits to the fragment no longer reached the prompt (1.89.0).
        """
        prompt = self._storage.load_prompt(name, version=version, alias=alias)
        if not resolve_fragments:
            return prompt

        resolved_template = self.resolve_fragments(prompt.template)
        if resolved_template != prompt.template:
            prompt = Prompt(
                name=prompt.name,
                template=resolved_template,
                version=prompt.version,
                metadata=prompt.metadata,
            )
        return prompt

    def _fetch_from_local(self, slug: str, version: int | None = None) -> Prompt:
        """Fetch prompt from local storage."""
        return self.load(slug, version=version)

    def _fetch_from_platform(self, slug: str, version: int | None = None) -> Prompt | None:
        """Fetch prompt from platform with TTL caching."""
        cache_key = (slug, version)

        # Check cache
        if cache_key in self._platform_cache:
            prompt, expires_at = self._platform_cache[cache_key]
            if time.monotonic() < expires_at:
                return prompt
            del self._platform_cache[cache_key]

        from fastaiagent._internal.errors import PlatformNotConnectedError

        try:
            from fastaiagent._platform.api import get_platform_api

            api = get_platform_api()
            params: dict[str, Any] = {}
            if version is not None:
                params["version"] = version
            data = api.get(f"/public/v1/prompts/{slug}", params=params or None)

            prompt = Prompt(
                name=data.get("slug", slug),
                template=data.get("content", ""),
                variables=data.get("variables", []),
                version=data.get("version", 1),
                metadata=data.get("metadata", {}),
                # Gap 4 provenance — retained so an agent using this prompt can
                # stamp fastaiagent.prompt.* on its llm_call span.
                slug=data.get("slug", slug),
                source="platform",
                environment=data.get("environment"),
            )

            # Cache with TTL
            self._platform_cache[cache_key] = (prompt, time.monotonic() + self._cache_ttl)
            return prompt
        except PlatformNotConnectedError:
            raise
        except Exception:
            logger.debug("Failed to fetch prompt from platform", exc_info=True)
            return None

    def _is_connected(self) -> bool:
        from fastaiagent.client import _connection

        return _connection.is_connected

    def resolve_fragments(self, template: str) -> str:
        """Replace {{@fragment_name}} with fragment content.

        An unknown fragment is left as written, so the gap stays visible."""

        def replacer(match: re.Match[str]) -> str:
            frag_name = match.group(1)
            # Check in-memory cache first
            if frag_name in self._fragments:
                return self._fragments[frag_name].content
            # Then try storage
            try:
                fragment = self._storage.load_fragment(frag_name)
                self._fragments[frag_name] = fragment
                return fragment.content
            except Exception:
                logger.debug("Failed to resolve prompt fragment %r", frag_name, exc_info=True)
                return match.group(0)  # leave unresolved

        result: str = _FRAGMENT_RE.sub(replacer, template)
        return result

    # Kept for callers of the pre-1.89.0 private name.
    _resolve_fragments = resolve_fragments

    def list(self) -> list[dict[str, Any]]:
        """List all registered prompts."""
        return self._storage.list_prompts()

    def diff(self, name: str, version_a: int, version_b: int) -> str:
        """A unified line diff between two versions' stored templates.

        Since 1.89.0 a line diff (``difflib.unified_diff``): it printed both
        whole templates as one ``-``/``+`` pair, so a one-word change in a long
        prompt meant comparing two pages by eye. ``{{@fragment}}`` references
        are compared as written.
        """
        import difflib

        a = self._storage.load_prompt(name, version=version_a)
        b = self._storage.load_prompt(name, version=version_b)

        header = [f"--- {name} v{version_a}", f"+++ {name} v{version_b}"]
        if a.template == b.template:
            return "\n".join([*header, "  (no template changes)"])
        body = list(
            difflib.unified_diff(
                a.template.splitlines(),
                b.template.splitlines(),
                lineterm="",
                n=2,
            )
        )
        # difflib's own ---/+++ lines name nothing; keep ours.
        return "\n".join([*header, *body[2:]])

    def set_alias(self, name: str, version: int, alias: str) -> None:
        """Set an alias (e.g., 'production') for a specific version.

        Raises :class:`PromptNotFoundError` when the prompt or the version does
        not exist (1.89.0; an alias to a missing version used to be accepted).
        """
        self._storage.set_alias(name, version, alias)

    def aliases(self, name: str) -> dict[str, int]:
        """``{alias: version}`` for a prompt (1.89.0)."""
        return self._storage.list_aliases(name)

    def delete_alias(self, name: str, alias: str) -> bool:
        """Remove an alias; ``False`` when it did not exist (1.89.0)."""
        return self._storage.delete_alias(name, alias)
