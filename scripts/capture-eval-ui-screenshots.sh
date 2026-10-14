#!/usr/bin/env bash
# 1.89.0 Phases 6–8 — eval numbers on screen and Run eval from the UI,
# asserted in the Local UI.
#
# Seeds a fresh local.db with two real evaluate() runs of a gpt-4o-mini support
# agent over a 5-row dataset (`contains` + an llm_judge): one with a working
# judge, gated on cases.pass_rate, and one whose judge model does not exist,
# gated on that judge. Starts `fastaiagent ui start --db … --no-auth --agent
# agent_app.py:agent` and drives Playwright through scripts/capture_eval_ui.py:
# cases vs checks, errored / n/a, the invalid gate, agent and eval cost,
# compare, and the Run eval dialog end to end. Writes
# docs/ui/screenshots/189-eval-*.png. Exit 1 on any failed assertion or
# console error.
#
# Live — needs OPENAI_API_KEY (it lives in ~/.zshrc), ~35 gpt-4o-mini calls:
#     zsh -c 'source ~/.zshrc; unset FASTAIAGENT_API_KEY FASTAIAGENT_TARGET; scripts/capture-eval-ui-screenshots.sh'

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_capture189_lib.sh"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY not set — run via: zsh -c 'source ~/.zshrc; scripts/capture-eval-ui-screenshots.sh'" >&2
  exit 1
fi

cap_setup 7874

echo "▸ seeding $LOCAL_DB (live gpt-4o-mini)"
"$PY" "$REPO_ROOT/scripts/capture_eval_ui.py" seed

cap_start_ui --agent "$TMP_ROOT/agent_app.py:agent"

echo "▸ driving the UI"
"$PY" "$REPO_ROOT/scripts/capture_eval_ui.py" drive "$BASE_URL"
