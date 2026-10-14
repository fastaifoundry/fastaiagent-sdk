#!/usr/bin/env bash
# 1.89.0 Phase 4 — replay isolation and provenance, asserted in the Local UI.
#
# Seeds a fresh local.db with two runs of a refund agent (a real FunctionTool
# with a side effect) and one recorded Python rerun, starts `fastaiagent ui
# start --db … --no-auth` (no --agent, so the UI has no implementation of the
# tool), and drives Playwright through scripts/capture_replay_isolation.py: the
# fork dialog's tool policy, the rerun's tool chips and warnings, the source
# badge and Hide experiments filter, and the experiments notes on Analytics,
# Home and Agents. Writes docs/ui/screenshots/189-replay-*.png. Exit 1 on any
# failed assertion or console error.
#
# Offline — no API key needed:
#     scripts/capture-replay-isolation.sh

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_capture189_lib.sh"

cap_setup 7872
cap_offline

echo "▸ seeding $LOCAL_DB"
"$PY" "$REPO_ROOT/scripts/capture_replay_isolation.py" seed

cap_start_ui

echo "▸ driving the UI"
"$PY" "$REPO_ROOT/scripts/capture_replay_isolation.py" drive "$BASE_URL" "$TMP_ROOT"
