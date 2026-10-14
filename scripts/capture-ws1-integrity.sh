#!/usr/bin/env bash
# 1.89.0 Phase 2 — data integrity, asserted in the Local UI.
#
# Seeds a fresh local.db (a prompt with a {{@fragment}} and metadata; three
# real pytest Agent-CI sessions: a baseline, a regression, a missing baseline),
# starts `fastaiagent ui start --db … --no-auth`, and drives Playwright through
# scripts/capture_ws1_integrity.py: the prompt editor, the recorded gate
# verdicts, the Playground's invalid-input checks and a stream that fails
# part-way. Writes docs/ui/screenshots/189-ws1-*.png. Exit 1 on any failed
# assertion or console error.
#
# Offline — no API key needed:
#     scripts/capture-ws1-integrity.sh
#     PORT=7881 KEEP_TMP=1 scripts/capture-ws1-integrity.sh

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_capture189_lib.sh"

cap_setup 7871
cap_offline

echo "▸ seeding $LOCAL_DB"
"$PY" "$REPO_ROOT/scripts/capture_ws1_integrity.py" seed

cap_dummy_openai_key
cap_start_ui

echo "▸ driving the UI"
"$PY" "$REPO_ROOT/scripts/capture_ws1_integrity.py" drive "$BASE_URL"
