#!/usr/bin/env bash
# Capture the Local UI screenshots for the AutoLLM closed-loop flagship (1.87.0).
#
# Runs examples/autollm-loop end to end against the live API (run_all.sh, ~10
# minutes), boots the UI on the example's own local.db, and drives Playwright
# through scripts/capture_autollm_loop_screenshots.py. Writes
# docs/ui/screenshots/autollm-loop-*.png.
#
# Needs OPENAI_API_KEY (it lives in ~/.zshrc):
#     zsh -lc 'scripts/capture-autollm-loop-screenshots.sh'
#     zsh -lc 'SKIP_RUN=1 scripts/capture-autollm-loop-screenshots.sh'   # reuse the last run

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
PORT="${PORT:-7863}"
EXAMPLE="$REPO_ROOT/examples/autollm-loop"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/docs/ui/screenshots}"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

if [ -z "${SKIP_RUN:-}" ]; then
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY not set — run via: zsh -lc 'scripts/capture-autollm-loop-screenshots.sh'" >&2
    exit 1
  fi
  echo "▸ running the loop live (examples/autollm-loop/run_all.sh)"
  PYTHON="$PY" "$EXAMPLE/run_all.sh" > "$TMP_ROOT/run_all.log"
  grep -E "^(baseline|holdout|best|gate|registered|triage_match)" "$TMP_ROOT/run_all.log" || true
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
# From the example folder, so the Dataset Editor reads its ./.fastaiagent/datasets.
# exec: the subshell becomes the server, so SERVER_PID is what the trap kills.
(cd "$EXAMPLE" && exec "$PY" "$TMP_ROOT/serve.py" .fastaiagent/local.db "$PORT") &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; rm -rf "$TMP_ROOT"' EXIT
for _ in $(seq 1 100); do
  curl -fsS -o /dev/null "http://127.0.0.1:$PORT/api/auth/status" 2>/dev/null && break
  sleep 0.2
done

echo "▸ capturing"
"$PY" "$REPO_ROOT/scripts/capture_autollm_loop_screenshots.py" "http://127.0.0.1:$PORT" "$OUT_DIR" "$EXAMPLE"
echo "▸ done — $OUT_DIR/autollm-loop-*.png"
