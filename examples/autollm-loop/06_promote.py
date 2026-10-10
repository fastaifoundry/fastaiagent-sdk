"""Step 6 — the winner becomes the next registry version. Production doesn't move.

The new version carries everything the winner is — its prompt as the template,
the worked examples it picked in the metadata — plus where it came from: the
AutoLLM run, and the dev and holdout scores it earned. Registering is not
shipping: 'production' still points at the old version until the gate passes.

UI: Prompts → ticket-triage (the new version, its metadata, a diff).
"""

from __future__ import annotations

import difflib

from triage import OUT, PROMPT, load_json, save_json

from fastaiagent.prompt import PromptRegistry


def main() -> None:
    run = load_json("optimize.json")
    (OUT / "promoted.json").unlink(missing_ok=True)  # never gate a stale promotion
    if not run["improved"]:
        print("AutoLLM found nothing that beat production on the holdout — nothing to promote.")
        return

    reg = PromptRegistry()
    base = reg.load(PROMPT, version=run["from_version"])
    winner = run["winner"]  # a lever left as None means "unchanged from the base"
    template = winner["system_prompt"] or base.template
    demos = winner["fewshot_demos"] or base.metadata.get("fewshot_demos")

    new = reg.register(
        PROMPT,
        template,
        metadata={
            "source": "autollm",
            "optimize_run_id": run["run_id"],
            "from_version": base.version,
            "dev": run["dev"],
            "holdout": run["holdout"],
            "fewshot_demos": demos,
        },
    )
    save_json("promoted.json", {"version": new.version, "from_version": base.version})

    print(
        f"registered {PROMPT!r} v{new.version} (from v{base.version}, AutoLLM run {run['run_id']})"
    )
    print(
        f"holdout {run['holdout']['baseline']:.3f} → {run['holdout']['best']:.3f}; "
        f"{len(demos or [])} worked examples\n"
    )
    for line in difflib.unified_diff(
        base.template.splitlines(),
        template.splitlines(),
        f"v{base.version}",
        f"v{new.version}",
        lineterm="",
    ):
        print(line)
    print("\n'production' still points at", f"v{reg.load(PROMPT, alias='production').version}.")


if __name__ == "__main__":
    main()
