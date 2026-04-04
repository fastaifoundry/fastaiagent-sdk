"""Dataset class for evaluation."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class Dataset:
    """A collection of test cases for evaluation.

    Example:
        ds = Dataset.from_jsonl("test_cases.jsonl")
        for item in ds:
            print(item["input"])
    """

    def __init__(self, items: list[dict[str, Any]]):
        self._items = items

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Dataset:
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
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
