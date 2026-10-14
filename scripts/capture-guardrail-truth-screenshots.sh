#!/usr/bin/env bash
# 1.89.0 Phase 5 — guardrails tell the truth, asserted in the Local UI.
#
# Seeds a fresh local.db by running an agent with four local regex rules (an
# input block, an output mask, a warn and an observe-only mask) with tracing
# on and FASTAIAGENT_UI_ENABLED unset, starts `fastaiagent ui start --db …
# --no-auth`, and drives Playwright through scripts/capture_guardrail_truth.py:
# the events list (effect, passes hidden), the event page (judged text, the
# rule as it ran, what happened next), run status from the root, and the
# agent's Guardrails tab. Writes docs/ui/screenshots/189-guardrail-*.png. Exit
# 1 on any failed assertion or console error.
#
# Offline — no API key needed:
#     scripts/capture-guardrail-truth-screenshots.sh

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_capture189_lib.sh"

cap_setup 7873
cap_offline

echo "▸ seeding $LOCAL_DB (FASTAIAGENT_UI_ENABLED unset)"
"$PY" "$REPO_ROOT/scripts/capture_guardrail_truth.py" seed

cap_start_ui

echo "▸ driving the UI"
"$PY" "$REPO_ROOT/scripts/capture_guardrail_truth.py" drive "$BASE_URL"
