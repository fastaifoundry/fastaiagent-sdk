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
