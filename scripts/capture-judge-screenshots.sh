#!/usr/bin/env bash
# Capture the Local UI screenshots for the judge-calibration page (1.87.0).
#
# Runs examples/autollm/calibrate_judge.py --tune-agent live (~5 minutes), boots
# the UI on that example's local.db, and drives Playwright through
# scripts/capture_judge_screenshots.py. Writes docs/ui/screenshots/judge-*.png.
#
# Needs OPENAI_API_KEY (it lives in ~/.zshrc):
#     zsh -lc 'scripts/capture-judge-screenshots.sh'
#     zsh -lc 'SKIP_RUN=1 scripts/capture-judge-screenshots.sh'   # reuse the last run

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
PORT="${PORT:-7864}"
EXAMPLE="$REPO_ROOT/examples/autollm"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/docs/ui/screenshots}"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

if [ -z "${SKIP_RUN:-}" ]; then
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY not set — run via: zsh -lc 'scripts/capture-judge-screenshots.sh'" >&2
    exit 1
  fi
  echo "▸ running calibrate_judge.py --tune-agent live (from a clean store)"
  rm -rf "$EXAMPLE/.fastaiagent" "$EXAMPLE/out"
  (cd "$EXAMPLE" && "$PY" calibrate_judge.py --tune-agent) | tee "$TMP_ROOT/judge.log" \
    | grep -E "^(best|holdout|  (naive|calibrated) judge)" || true
fi

if lsof -ti ":$PORT" >/dev/null 2>&1; then
  echo "port $PORT is busy — set PORT=<free port>" >&2
  exit 1
fi

echo "▸ booting the UI on :$PORT (no-auth) over the example's local.db"
cat > "$TMP_ROOT/serve.py" <<'PY'
import sys
import uvicorn
from fastaiagent.ui.server import build_app

app = build_app(db_path=sys.argv[1], no_auth=True)
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[2]), log_level="warning")
PY
# exec: the subshell becomes the server, so SERVER_PID is what the trap kills.
(cd "$EXAMPLE" && exec "$PY" "$TMP_ROOT/serve.py" .fastaiagent/local.db "$PORT") &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; rm -rf "$TMP_ROOT"' EXIT
for _ in $(seq 1 100); do
  curl -fsS -o /dev/null "http://127.0.0.1:$PORT/api/auth/status" 2>/dev/null && break
  sleep 0.2
done

echo "▸ capturing"
"$PY" "$REPO_ROOT/scripts/capture_judge_screenshots.py" "http://127.0.0.1:$PORT" "$OUT_DIR"
echo "▸ done — $OUT_DIR/judge-*.png"
