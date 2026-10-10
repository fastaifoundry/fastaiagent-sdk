"""Step 1 — v1 of the triage prompt goes into the registry, and goes live.

The agent never holds its prompt in code: every step builds it from the
registry's ``production`` alias. So "ship a better prompt" later is one alias
move, and every trace says which version produced it.

    python 01_register_prompt.py           # keeps an existing prompt
    python 01_register_prompt.py --reset   # start the loop over

UI: Prompts → ticket-triage.
"""

from __future__ import annotations

import argparse
import shutil

from triage import OUT, PROMPT, V1_PROMPT, dataset_path

from fastaiagent.prompt import PromptRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="delete every version first")
    args = parser.parse_args()

    reg = PromptRegistry()
    if args.reset:
        removed = reg.delete(PROMPT)
        dataset_path().unlink(missing_ok=True)
        shutil.rmtree(OUT, ignore_errors=True)
        print(f"reset: removed {removed} version(s) of {PROMPT!r}, the dataset and out/")
        print("       (traces and eval runs stay; delete .fastaiagent/ for a clean slate)")

    if any(p["name"] == PROMPT for p in reg.list()):
        live = reg.load(PROMPT, alias="production")
        print(f"{PROMPT!r} is already registered — production is v{live.version}")
        return

    v1 = reg.register(PROMPT, V1_PROMPT, metadata={"source": "hand-written"})
    reg.set_alias(PROMPT, v1.version, "production")
    print(f"registered {PROMPT!r} v{v1.version} and pointed 'production' at it\n")
    print(V1_PROMPT)


if __name__ == "__main__":
    main()
