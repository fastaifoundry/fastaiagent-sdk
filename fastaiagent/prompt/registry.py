"""Prompt registry with versioning, aliases, fragment composition, and platform support."""

from __future__ import annotations

import re
import time
from typing import Any

from fastaiagent.prompt.fragment import Fragment
from fastaiagent.prompt.prompt import Prompt
from fastaiagent.prompt.storage import YAMLStorage

_DEFAULT_CACHE_TTL = 300  # 5 minutes


class PromptRegistry:
    """Prompt registry with local file storage and optional platform support.

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

    def __init__(self, store: str = "local", path: str = ".prompts/"):
        self._storage = YAMLStorage(path=path)
        self._fragments: dict[str, Fragment] = {}
        self._platform_cache: dict[tuple[str, int | None], tuple[Prompt, float]] = {}
        self._cache_ttl: int = _DEFAULT_CACHE_TTL

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
    ) -> Prompt:
        """Register a new prompt (creates a new version)."""
        try:
            existing = self._storage.load_prompt(name)
            new_version = existing.version + 1
        except Exception:
            new_version = 1

        if version is not None:
            new_version = version

        prompt = Prompt(
            name=name,
            template=template,
            version=new_version,
            metadata=metadata or {},
        )
        self._storage.save_prompt(prompt)
        return prompt

    def register_fragment(self, name: str, content: str) -> Fragment:
        """Register a reusable prompt fragment."""
        fragment = Fragment(name=name, content=content)
        self._storage.save_fragment(fragment)
        self._fragments[name] = fragment
        return fragment

    def load(self, name: str, version: int | None = None, alias: str | None = None) -> Prompt:
        """Load a prompt from local storage, resolving {{@fragment}} references."""
        prompt = self._storage.load_prompt(name, version=version, alias=alias)

        # Resolve fragment references
        resolved_template = self._resolve_fragments(prompt.template)
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
            )

            # Cache with TTL
            self._platform_cache[cache_key] = (prompt, time.monotonic() + self._cache_ttl)
            return prompt
        except PlatformNotConnectedError:
            raise
        except Exception:
            return None

    def _is_connected(self) -> bool:
        from fastaiagent.client import _connection

        return _connection.is_connected

    def _resolve_fragments(self, template: str) -> str:
        """Replace {{@fragment_name}} with fragment content."""

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
                return match.group(0)  # leave unresolved

        result: str = re.sub(r"\{\{@(\w+)\}\}", replacer, template)
        return result

    def list(self) -> list[dict[str, Any]]:
        """List all registered prompts."""
        return self._storage.list_prompts()

    def diff(self, name: str, version_a: int, version_b: int) -> str:
        """Show diff between two versions of a prompt."""
        a = self._storage.load_prompt(name, version=version_a)
        b = self._storage.load_prompt(name, version=version_b)

        lines = [
            f"--- {name} v{version_a}",
            f"+++ {name} v{version_b}",
        ]
        if a.template != b.template:
            lines.append(f"- {a.template}")
            lines.append(f"+ {b.template}")
        else:
            lines.append("  (no template changes)")
        return "\n".join(lines)

    def set_alias(self, name: str, version: int, alias: str) -> None:
        """Set an alias (e.g., 'production') for a specific version."""
        self._storage.set_alias(name, version, alias)
