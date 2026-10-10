#!/usr/bin/env bash
# The whole loop, in order, from a clean slate. Needs OPENAI_API_KEY:
#
#     zsh -lc './run_all.sh'        # if your key lives in ~/.zshrc
#
# About 10 minutes — most of it AutoLLM (step 5) and the CI gate (step 7).
# Starts by deleting THIS folder's .fastaiagent/ and out/ (the example's own
# store), so the UI shows exactly one pass of the loop. Then:
#
#     fastaiagent ui           # from this folder — it reads ./.fastaiagent/local.db

set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

rm -rf .fastaiagent out
"$PY" 01_register_prompt.py
"$PY" 02_serve_traffic.py
"$PY" 03_curate.py
"$PY" 04_baseline.py
"$PY" 05_optimize.py
"$PY" 06_promote.py
"$PY" 07_go_live.py
"$PY" 02_serve_traffic.py
