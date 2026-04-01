"""Prompt registry with versioning, aliases, and fragment composition."""

from __future__ import annotations

import re
from typing import Any

from fastaiagent.prompt.fragment import Fragment
from fastaiagent.prompt.prompt import Prompt
from fastaiagent.prompt.storage import YAMLStorage


class PromptRegistry:
    """Local file-based prompt registry.

    Example:
        reg = PromptRegistry()
        reg.register_fragment(name="tone", content="Be professional.")
        reg.register(name="greeting", template="Hello {{name}}. {{@tone}}")
        prompt = reg.load("greeting")
        text = prompt.format(name="World")
    """

    def __init__(self, store: str = "local", path: str = ".prompts/"):
        self._storage = YAMLStorage(path=path)
        self._fragments: dict[str, Fragment] = {}

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
        """Load a prompt, resolving {{@fragment}} references."""
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
