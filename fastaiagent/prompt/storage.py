"""YAML-based local prompt storage."""

from __future__ import annotations

import json
from pathlib import Path

from fastaiagent._internal.errors import PromptNotFoundError
from fastaiagent.prompt.fragment import Fragment
from fastaiagent.prompt.prompt import Prompt


class YAMLStorage:
    """File-based prompt storage using JSON (YAML-like structure)."""

    def __init__(self, path: str = ".prompts/"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def save_prompt(self, prompt: Prompt) -> None:
        file = self.path / f"{prompt.name}.json"
        existing = self._load_file(file)
        versions = existing.get("versions", [])
        versions.append(prompt.to_dict())
        existing["name"] = prompt.name
        existing["versions"] = versions
        existing["latest_version"] = prompt.version
        if "aliases" not in existing:
            existing["aliases"] = {}
        file.write_text(json.dumps(existing, indent=2))

    def load_prompt(
        self, name: str, version: int | None = None, alias: str | None = None
    ) -> Prompt:
        file = self.path / f"{name}.json"
        if not file.exists():
            raise PromptNotFoundError(f"Prompt '{name}' not found")
        data = self._load_file(file)
        versions = data.get("versions", [])
        if not versions:
            raise PromptNotFoundError(f"Prompt '{name}' has no versions")

        if alias:
            target_version = data.get("aliases", {}).get(alias)
            if target_version is None:
                raise PromptNotFoundError(f"Alias '{alias}' not found for prompt '{name}'")
            version = target_version

        if version is not None:
            for v in versions:
                if v.get("version") == version:
                    return Prompt.from_dict(v)
            raise PromptNotFoundError(f"Version {version} not found for prompt '{name}'")

        return Prompt.from_dict(versions[-1])

    def set_alias(self, name: str, version: int, alias: str) -> None:
        file = self.path / f"{name}.json"
        if not file.exists():
            raise PromptNotFoundError(f"Prompt '{name}' not found")
        data = self._load_file(file)
        data.setdefault("aliases", {})[alias] = version
        file.write_text(json.dumps(data, indent=2))

    def save_fragment(self, fragment: Fragment) -> None:
        file = self.path / f"_fragment_{fragment.name}.json"
        file.write_text(json.dumps(fragment.to_dict(), indent=2))

    def load_fragment(self, name: str) -> Fragment:
        file = self.path / f"_fragment_{name}.json"
        if not file.exists():
            from fastaiagent._internal.errors import FragmentNotFoundError

            raise FragmentNotFoundError(f"Fragment '{name}' not found")
        return Fragment.from_dict(json.loads(file.read_text()))

    def list_prompts(self) -> list[dict]:
        results = []
        for file in sorted(self.path.glob("*.json")):
            if file.name.startswith("_fragment_"):
                continue
            data = self._load_file(file)
            results.append({
                "name": data.get("name", file.stem),
                "latest_version": data.get("latest_version", 1),
                "versions": len(data.get("versions", [])),
            })
        return results

    def _load_file(self, file: Path) -> dict:
        if not file.exists():
            return {}
        return json.loads(file.read_text())
