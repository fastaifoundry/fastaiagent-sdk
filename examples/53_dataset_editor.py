"""Example 53 — Eval Dataset Editor demo (Sprint 3).

Seeds two datasets in ``.fastaiagent/datasets/``:

  * ``echo-strict`` — 3 plain-text cases (``exact_match`` scorer).
  * ``vision-smoke`` — 2 multimodal cases referencing a checked-in
    image fixture, demonstrating the typed-parts shape.

After running, point the Local UI at the project and the editor lists
both datasets immediately. No LLM call is made — the example is purely
filesystem seeding so it's safe to re-run, and free.

Prereqs:
    pip install 'fastaiagent[ui]'

Run:
    python examples/53_dataset_editor.py
    fastaiagent ui --no-auth
    # Open http://127.0.0.1:7842/datasets
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63fcffff3f0300055e02fea33cc7ab0000000049454e44ae426082"
)


def _datasets_dir() -> Path:
    base = Path.cwd() / ".fastaiagent" / "datasets"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    base = _datasets_dir()

    # --- Plain text dataset
    echo_path = base / "echo-strict.jsonl"
    _write_jsonl(
        echo_path,
        [
            {
                "input": "Reply with exactly the word 'ready'.",
                "expected_output": "ready",
                "tags": ["exact_match"],
                "metadata": {},
            },
            {
                "input": "Reply with exactly the word 'yes'.",
                "expected_output": "yes",
                "tags": ["exact_match"],
                "metadata": {},
            },
            {
                "input": "Reply with exactly the word 'no'.",
                "expected_output": "no",
                "tags": ["exact_match"],
                "metadata": {},
            },
        ],
    )

    # --- Multimodal dataset
    vision_dir = base / "images" / "vision-smoke"
    vision_dir.mkdir(parents=True, exist_ok=True)
    img1 = vision_dir / "tile-a.png"
    img2 = vision_dir / "tile-b.png"
    img1.write_bytes(PNG_BYTES)
    img2.write_bytes(PNG_BYTES)
    vision_path = base / "vision-smoke.jsonl"
    _write_jsonl(
        vision_path,
        [
            {
                "input": [
                    {"type": "text", "text": "What is shown?"},
                    {"type": "image", "path": "images/vision-smoke/tile-a.png"},
                ],
                "expected_output": "blank",
                "tags": ["vision"],
                "metadata": {"source": "demo"},
            },
            {
                "input": [
                    {"type": "text", "text": "Compare the two images."},
                    {"type": "image", "path": "images/vision-smoke/tile-a.png"},
                    {"type": "image", "path": "images/vision-smoke/tile-b.png"},
                ],
                "expected_output": "identical",
                "tags": ["vision"],
                "metadata": {"source": "demo"},
            },
        ],
    )

    print(f"Seeded: {echo_path.relative_to(Path.cwd())}  ({3} cases)")
    print(f"Seeded: {vision_path.relative_to(Path.cwd())}  ({2} cases)")
    print()
    print("Open the editor:")
    print("  fastaiagent ui --no-auth")
    print("  http://127.0.0.1:7842/datasets")
    print()
    print("Each dataset can be edited inline, exported as JSONL, or run as")
    print("an eval (Run eval button → /evals/<run_id>). Multimodal cases")
    print("preview images side-by-side in the editor modal.")

    # Sanity: shutil exists so the linter doesn't complain about the
    # unused import — real cleanup is the user's call (delete from the
    # UI's trash icon if you want a clean slate).
    _ = shutil
    return 0


if __name__ == "__main__":
    sys.exit(main())
