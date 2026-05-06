"""
Evaluation Suite — quality testing for the meeting-notes pipeline.

Run: python eval_suite.py
     python eval_suite.py --publish     # publish to FastAIAgent Platform

Three custom Python scorers (no LLM-judge needed — the assertions are
deterministic against ground-truth labels):

  * **action_item_recall**  — every action in the must_extract list
    must appear in the output. Recall against a curated golden set.

  * **decision_recall**     — same, for decisions.

  * **owner_attribution**   — every action item must have a single
    named person as owner (no "the team", "we", "everyone"). The most
    common failure mode for action-item extraction.

Each case auto-runs the chain on a fixture transcript and grades the
resulting MeetingNotes against the labels. Persisted to local.db so the
run shows up at /evals in the Local UI.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa
from fastaiagent.eval.results import EvalCaseRecord, EvalResults
from fastaiagent.eval.scorer import ScorerResult

from schema import MeetingNotes
from tools import make_deps
from workflow import build_chain


_HERE = Path(__file__).resolve().parent

# ─── Eval dataset ────────────────────────────────────────────────────────────

EVAL_CASES: list[dict] = [
    {
        "input": str(_HERE / "fixtures" / "sample_transcript.md"),
        # Substrings — every one must appear in some action_item.text.
        "must_extract_actions": [
            "auth-middleware",
            "trace inspector",
            "SDETs",
        ],
        # Substrings — every one must appear in some decision.text.
        "must_extract_decisions": [
            "Slack",
            "auth",
        ],
        # Names that MUST appear as action-item owners (single-person attribution).
        "expected_owners": {"Bob Patel", "Carol Johnson", "Alice Chen"},
    },
]


# ─── Scorers ────────────────────────────────────────────────────────────────


def _score_action_recall(notes: MeetingNotes, must_extract: list[str]) -> tuple[float, str]:
    if not must_extract:
        return 1.0, "no required actions"
    bodies = [a.text.lower() for a in notes.action_items]
    hits = [needle for needle in must_extract if any(needle.lower() in b for b in bodies)]
    ratio = len(hits) / len(must_extract)
    return ratio, f"matched {len(hits)}/{len(must_extract)} required action substrings"


def _score_decision_recall(notes: MeetingNotes, must_extract: list[str]) -> tuple[float, str]:
    if not must_extract:
        return 1.0, "no required decisions"
    bodies = [d.text.lower() for d in notes.decisions]
    hits = [needle for needle in must_extract if any(needle.lower() in b for b in bodies)]
    ratio = len(hits) / len(must_extract)
    return ratio, f"matched {len(hits)}/{len(must_extract)} required decision substrings"


def _score_owner_attribution(notes: MeetingNotes, expected_owners: set[str]) -> tuple[float, str]:
    """Penalize team-attribution and missed-owner action items."""
    if not notes.action_items:
        return 0.0, "no action items extracted at all"
    forbidden_tokens = {"team", "we", "everyone", "all", ""}
    # Match at the word level so "the team", "team", "team will" all trip
    # the filter — but a person whose first name is "Allen" doesn't.
    def _is_team_attribution(owner: str) -> bool:
        words = set(owner.strip().lower().split()) if owner else {""}
        return bool(words & forbidden_tokens)

    well_attributed = [a for a in notes.action_items if not _is_team_attribution(a.owner)]
    saw_expected = {a.owner for a in well_attributed} & expected_owners
    # Score = (fraction well-attributed) * (fraction of expected owners covered)
    attribution = len(well_attributed) / len(notes.action_items)
    coverage = len(saw_expected) / max(len(expected_owners), 1)
    score = round(attribution * coverage, 4)
    return score, (
        f"{len(well_attributed)}/{len(notes.action_items)} actions named-owner "
        f"+ {len(saw_expected)}/{len(expected_owners)} expected owners covered"
    )


# ─── Runner ─────────────────────────────────────────────────────────────────


async def run_eval(publish: bool = False) -> None:
    print("\nPhase 1 — running chain on each fixture...\n")
    artifacts: dict[str, MeetingNotes | None] = {}
    for case in EVAL_CASES:
        path = case["input"]
        chain = build_chain()
        deps = make_deps()
        ctx = fa.RunContext(state=deps)
        execution_id = f"meeting-eval-{uuid.uuid4().hex[:8]}"
        try:
            result = await chain.aexecute(
                {"path": path}, execution_id=execution_id, context=ctx
            )
        except Exception as e:
            artifacts[path] = None
            print(f"  • {Path(path).name:<35} ERROR: {e}")
            continue
        notes_dict = result.final_state.get("output") or {}
        try:
            notes_obj = MeetingNotes.model_validate(notes_dict)
            artifacts[path] = notes_obj
            print(
                f"  • {Path(path).name:<35} actions={len(notes_obj.action_items)} "
                f"decisions={len(notes_obj.decisions)}"
            )
        except Exception as e:
            artifacts[path] = None
            print(f"  • {Path(path).name:<35} validation failed: {e}")

    print("\nPhase 2 — scoring...\n")
    results = EvalResults()
    for case in EVAL_CASES:
        path = case["input"]
        notes = artifacts.get(path)
        if notes is None:
            for scorer_name in ("action_item_recall", "decision_recall", "owner_attribution"):
                results.add(scorer_name, ScorerResult(score=0.0, passed=False, reason="chain failed"))
            continue

        ar_score, ar_reason = _score_action_recall(notes, case["must_extract_actions"])
        dr_score, dr_reason = _score_decision_recall(notes, case["must_extract_decisions"])
        oa_score, oa_reason = _score_owner_attribution(notes, set(case["expected_owners"]))

        print(f"  {Path(path).name:<35} ar={ar_score:.2f}  dr={dr_score:.2f}  oa={oa_score:.2f}")

        results.add("action_item_recall", ScorerResult(score=ar_score, passed=ar_score >= 1.0, reason=ar_reason))
        results.add("decision_recall", ScorerResult(score=dr_score, passed=dr_score >= 1.0, reason=dr_reason))
        results.add("owner_attribution", ScorerResult(score=oa_score, passed=oa_score >= 0.7, reason=oa_reason))
        results.add_case(
            EvalCaseRecord(
                input=path,
                actual_output=notes.model_dump_json(indent=2),
                per_scorer={
                    "action_item_recall": {"score": ar_score, "passed": ar_score >= 1.0, "reason": ar_reason},
                    "decision_recall": {"score": dr_score, "passed": dr_score >= 1.0, "reason": dr_reason},
                    "owner_attribution": {"score": oa_score, "passed": oa_score >= 0.7, "reason": oa_reason},
                },
            )
        )

    print()
    print(results.summary())

    try:
        run_id = results.persist_local(
            run_name="meeting-notes eval",
            dataset_name="meeting-fixtures-golden",
            agent_name="meeting-notes",
        )
        print(f"\n  ✓ persisted to local.db (run_id={run_id[:12]}...)")
    except Exception as e:
        print(f"\n  Could not persist eval run locally: {e}")

    if publish:
        try:
            results.publish(run_name="meeting-notes eval")
            print("  ✓ published to FastAIAgent Platform")
        except Exception as e:
            print(f"  Platform publish failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true", help="Publish to platform")
    args = parser.parse_args()
    asyncio.run(run_eval(publish=args.publish))


if __name__ == "__main__":
    main()
