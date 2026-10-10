"""Step 7 — gate the new version, and point 'production' at it only if it passes.

Runs ``test_gate.py`` exactly as CI would. A green gate moves the registry alias —
the one change that ships the prompt: every agent built from
``alias="production"`` picks it up on its next build, and no code is deployed.
A red gate leaves production where it was.

Then run step 2 again: the new traffic is stamped with the new version.

UI: Eval Runs → the gate's run (compare it with v1) · Prompts → ticket-triage.
"""

from __future__ import annotations

import sys

import pytest
from triage import OUT, PROMPT, load_json

from fastaiagent.prompt import PromptRegistry

PASS_BAR = "overall.pass_rate=0.85"


def main() -> int:
    if not (OUT / "promoted.json").exists():
        print("step 6 promoted nothing — 'production' stays where it is.")
        return 0
    candidate = load_json("promoted.json")["version"]
    baseline = load_json("baseline.json")["run_name"]

    print(f"gating {PROMPT!r} v{candidate}: bar {PASS_BAR}, no regression vs {baseline!r}\n")
    exit_code = pytest.main(
        [
            "test_gate.py",
            "-q",
            "-p",
            "no:randomly",
            "--eval-run-name",
            f"ticket-triage v{candidate} (gate)",
            "--eval-fail-under",
            PASS_BAR,
            "--eval-baseline",
            baseline,
            "--eval-tolerance",
            "0",
        ]
    )

    reg = PromptRegistry()
    if exit_code != 0:
        live = reg.load(PROMPT, alias="production").version
        print(f"\ngate FAILED — 'production' stays on v{live}.")
        return 1
    reg.set_alias(PROMPT, candidate, "production")
    print(f"\ngate passed — 'production' now points at v{candidate}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
