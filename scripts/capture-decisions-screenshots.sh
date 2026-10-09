#!/usr/bin/env bash
# Capture the Local UI screenshots for the OpenAI Decisions API docs (1.84.0).
#
# Runs the real examples against the live API — 105 (replay), 106 (call-centre
# Supervisor routed by the Decisions API), 107 (the same desk as a Chain) — into
# a throwaway local.db, boots the UI on it, and drives Playwright through
# scripts/capture_decisions_screenshots.py. Writes docs/ui/screenshots/decisions-*.png.
#
# Needs OPENAI_API_KEY (it lives in ~/.zshrc):
#     zsh -lc 'scripts/capture-decisions-screenshots.sh'
#
# Everything runs from a temp working directory: the repo's .fastaiagent/ holds
# legacy stores that the UI would otherwise import into the screenshot database.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
PORT="${PORT:-7849}"
TMP_ROOT="$(mktemp -d)"
LOCAL_DB="$TMP_ROOT/local.db"
OUT_DIR="$REPO_ROOT/docs/ui/screenshots"
trap 'rm -rf "$TMP_ROOT"' EXIT

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY not set — run via: zsh -lc 'scripts/capture-decisions-screenshots.sh'" >&2
  exit 1
fi

cd "$TMP_ROOT"
export FASTAIAGENT_LOCAL_DB="$LOCAL_DB"
echo "▸ running examples 106, 107, 105 live into $LOCAL_DB"
"$PY" "$REPO_ROOT/examples/106_call_center_supervisor.py" > "$TMP_ROOT/106.log"
"$PY" "$REPO_ROOT/examples/107_call_center_chain.py" > "$TMP_ROOT/107.log"
"$PY" "$REPO_ROOT/examples/105_decision_replay.py" > "$TMP_ROOT/105.log"
grep -E "^> |→" "$TMP_ROOT/106.log" | head -10 || true

if lsof -ti ":$PORT" >/dev/null 2>&1; then
  echo "port $PORT is busy — set PORT=<free port>" >&2
  exit 1
fi

echo "▸ booting the UI on :$PORT (no-auth)"
cat > "$TMP_ROOT/serve.py" <<'PY'
import sys
import uvicorn
from fastaiagent.ui.server import build_app

app = build_app(db_path=sys.argv[1], no_auth=True)
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[2]), log_level="warning")
PY
"$PY" "$TMP_ROOT/serve.py" "$LOCAL_DB" "$PORT" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; rm -rf "$TMP_ROOT"' EXIT
for _ in $(seq 1 100); do
  curl -fsS -o /dev/null "http://127.0.0.1:$PORT/api/auth/status" 2>/dev/null && break
  sleep 0.2
done

echo "▸ capturing"
"$PY" "$REPO_ROOT/scripts/capture_decisions_screenshots.py" "http://127.0.0.1:$PORT" "$OUT_DIR"
echo "▸ done — $OUT_DIR/decisions-*.png"
