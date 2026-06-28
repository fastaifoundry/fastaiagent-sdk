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
        # Set by :meth:`from_traces` to the ``CuratedDataset`` it built, so callers
        # can read curation coverage (e.g. ``ds.curation.infra_excluded`` /
        # ``ds.curation.coverage_summary()``). ``None`` for datasets from other
        # sources (jsonl/csv/list).
        self.curation: Any = None

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

    @classmethod
    def from_traces(cls, **kwargs: Any) -> Dataset:
        """Build a Dataset from captured agent traces in the local DB.

        Thin wrapper over :func:`fastaiagent.eval.curate.curate_from_traces`. Each
        ``agent.<name>`` span (root or nested in a chain/supervisor/swarm) becomes
        one item. See ``curate_from_traces`` for the keyword arguments
        (``filter``, ``agent``, ``since_hours``, ``limit``, ``trace_ids``,
        ``mark_output_as_expected``, ``db_path``, ``dedup_by``).

        Example:
            ds = Dataset.from_traces(filter="favorites")
            ds.to_jsonl("cases.jsonl")
        """
        from fastaiagent.eval.curate import curate_from_traces

        curated = curate_from_traces(**kwargs)
        ds = cls(curated)
        ds.curation = curated  # coverage: .infra_excluded / .emitted / .coverage_summary()
        return ds

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._items[idx]

    def to_jsonl(self, path: str | Path, *, append: bool = False) -> Path:
        """Write items as JSONL (one compact JSON object per line).

        Uses the same line format as ``ReplayResult.save_as_test`` so curated and
        replay-saved cases interleave in one file and round-trip through
        :meth:`from_jsonl`. Pass ``append=True`` to add to an existing file.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a" if append else "w", encoding="utf-8") as f:
            for item in self._items:
                f.write(json.dumps(item, default=str) + "\n")
        return p

    @classmethod
    def from_platform(cls, name: str) -> Dataset:
        """Pull dataset from platform."""
        from fastaiagent._internal.errors import PlatformNotConnectedError
        from fastaiagent._platform.api import get_platform_api
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            raise PlatformNotConnectedError("Not connected to platform. Call fa.connect() first.")
        api = get_platform_api()
        data = api.get(f"/public/v1/eval/datasets/{name}")
        return cls(data.get("items", []))

    def publish(self, name: str) -> None:
        """Push dataset to platform."""
        from fastaiagent._internal.errors import PlatformNotConnectedError
        from fastaiagent._platform.api import get_platform_api
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            raise PlatformNotConnectedError("Not connected to platform. Call fa.connect() first.")
        api = get_platform_api()
        api.post(
            "/public/v1/eval/datasets",
            {"name": name, "items": self._items},
        )
