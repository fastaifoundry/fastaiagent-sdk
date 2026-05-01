"""Dataset class for evaluation."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _resolve_multimodal_part(part: Any, base_dir: Path) -> Any:
    """Convert a JSONL part dict into a real ``Image``/``PDF`` instance.

    Supported shapes (paths are relative to ``base_dir``)::

        {"type": "text", "text": "..."}        -> str
        {"type": "image", "path": "x.jpg"}     -> Image.from_file(...)
        {"type": "image", "url":  "https://..."}  -> Image.from_url(...)
        {"type": "pdf",   "path": "x.pdf"}     -> PDF.from_file(...)
        {"type": "pdf",   "url":  "https://..."}  -> PDF.from_url(...)

    Anything else passes through unchanged so existing eval datasets are
    fully backward-compatible.
    """
    if not isinstance(part, dict):
        return part
    kind = part.get("type")
    if kind == "text":
        return part.get("text", "")
    if kind == "image":
        from fastaiagent.multimodal.image import Image

        if "path" in part:
            resolved = (base_dir / part["path"]).resolve()
            return Image.from_file(resolved, detail=part.get("detail", "auto"))
        if "url" in part:
            return Image.from_url(part["url"], detail=part.get("detail", "auto"))
    if kind == "pdf":
        from fastaiagent.multimodal.pdf import PDF

        if "path" in part:
            resolved = (base_dir / part["path"]).resolve()
            return PDF.from_file(resolved)
        if "url" in part:
            return PDF.from_url(part["url"])
    return part


def _resolve_multimodal_input(value: Any, base_dir: Path) -> Any:
    """Walk an ``input`` field and resolve any image/pdf markers it contains."""
    if isinstance(value, list):
        return [_resolve_multimodal_part(p, base_dir) for p in value]
    return _resolve_multimodal_part(value, base_dir)


class Dataset:
    """A collection of test cases for evaluation.

    Example:
        ds = Dataset.from_jsonl("test_cases.jsonl")
        for item in ds:
            print(item["input"])

    Multimodal items are supported in ``from_jsonl``: an item's ``input``
    can be a list of typed parts, e.g.::

        {"input": [
            {"type": "text", "text": "What letters appear?"},
            {"type": "image", "path": "fixtures/cat.jpg"}
        ], "expected": "CAT"}

    Paths are resolved relative to the JSONL file's directory and turn
    into ``Image``/``PDF`` instances at load time.
    """

    def __init__(self, items: list[dict[str, Any]]):
        self._items = items

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Dataset:
        base_dir = Path(path).resolve().parent
        items: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if "input" in raw:
                    raw["input"] = _resolve_multimodal_input(raw["input"], base_dir)
                items.append(raw)
        return cls(items)

    @classmethod
    def from_csv(cls, path: str | Path) -> Dataset:
        items = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(dict(row))
        return cls(items)

    @classmethod
    def from_list(cls, items: list[dict[str, Any]]) -> Dataset:
        return cls(items)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dataset:
        return cls(data.get("items", []))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._items[idx]

    @classmethod
    def from_platform(cls, name: str) -> Dataset:
        """Pull dataset from platform."""
        from fastaiagent._internal.errors import PlatformNotConnectedError
        from fastaiagent._platform.api import get_platform_api
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            raise PlatformNotConnectedError(
                "Not connected to platform. Call fa.connect() first."
            )
        api = get_platform_api()
        data = api.get(f"/public/v1/eval/datasets/{name}")
        return cls(data.get("items", []))

    def publish(self, name: str) -> None:
        """Push dataset to platform."""
        from fastaiagent._internal.errors import PlatformNotConnectedError
        from fastaiagent._platform.api import get_platform_api
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            raise PlatformNotConnectedError(
                "Not connected to platform. Call fa.connect() first."
            )
        api = get_platform_api()
        api.post(
            "/public/v1/eval/datasets",
            {"name": name, "items": self._items},
        )
