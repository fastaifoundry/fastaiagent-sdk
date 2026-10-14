#!/usr/bin/env bash
# 1.89.0 Phase 9 — Playground, prompts, approvals, KB and traces, asserted in
# the Local UI.
#
# Seeds a fresh local.db (a prompt with two versions and two aliases, a
# keyword-only LocalKB, and a refund agent paused on a manager approval),
# starts `fastaiagent ui start --db … --no-auth --agent agent_app.py:agent`, and
# drives Playwright through scripts/capture_playground_fixes.py against a
# localhost SSE stub scripted per run: Playground errors and hints, sampling
# rules, endpoint-required providers, unfilled variables, truncation, aliases
# and attribution, history restore, the prompt Compare panel, Playground and
# Retrieval badges, trace notes and the Noted filter, KB search type, and
# resolving the approval into the execution Outcome card. Writes
# docs/ui/screenshots/189-playground-*.png. Exit 1 on any failed assertion or
# console error.
#
# Offline — no API key needed:
#     scripts/capture-playground-fixes-screenshots.sh

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_capture189_lib.sh"

cap_setup 7875
cap_offline

echo "▸ seeding $LOCAL_DB"
"$PY" "$REPO_ROOT/scripts/capture_playground_fixes.py" seed

cap_dummy_openai_key
cap_start_ui --agent "$TMP_ROOT/agent_app.py:agent"

echo "▸ driving the UI"
"$PY" "$REPO_ROOT/scripts/capture_playground_fixes.py" drive "$BASE_URL"
