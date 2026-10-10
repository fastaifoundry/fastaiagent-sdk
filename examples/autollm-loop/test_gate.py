"""The CI gate for a candidate prompt version — plain pytest, one test per ticket.

Each labelled ticket is a test case; the fastaiagent pytest plugin gathers them
into ONE persisted eval run and gates the aggregate. ``07_go_live.py`` runs it
like this:

    pytest test_gate.py -q \\
        --eval-run-name "ticket-triage v2 (gate)" \\
        --eval-fail-under "overall.pass_rate=0.85" \\
        --eval-baseline "ticket-triage v1" --eval-tolerance 0

``--eval-fail-under`` is the bar; ``--eval-baseline`` fails the build if the
candidate's pass rate drops below the version that's live. A ticket the
candidate gets wrong stays a green test (``assert_pass=False``): the gate judges
the whole set, which is what decides a release.

The dataset includes the cases AutoLLM trained on, so this is a regression check
— the generalisation number is the holdout score from step 5.

The version under test: ``TRIAGE_GATE_VERSION``, else the one step 6 promoted.
"""

from __future__ import annotations

import os

from triage import TriageMatch, dataset_path, load_agent, load_json

from fastaiagent.eval import pytest_dataset

VERSION = int(os.environ.get("TRIAGE_GATE_VERSION") or load_json("promoted.json")["version"])
AGENT = load_agent(version=VERSION)


@pytest_dataset(str(dataset_path()))
def test_ticket(eval_case, evaluate_one):  # type: ignore[no-untyped-def]
    evaluate_one(AGENT.run, scorers=[TriageMatch()], assert_pass=False)
