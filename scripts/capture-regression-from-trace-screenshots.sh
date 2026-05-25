#!/usr/bin/env bash
# Capture before/after screenshots of the regression-from-trace
# template's failing and fixed runs, for embedding in
# docs/flagships/regression-from-trace.md and the example README.
#
# What this does:
#   1. Points FASTAIAGENT_HOME at a tmpdir so the example's traces
#      go to a scoped local.db (the user's home DB is untouched).
#   2. Runs capture.py + fix.py to produce a failing trace and a
#      fixed rerun trace inside that DB.
#   3. Reads back both trace IDs from the DB.
#   4. Boots the FastAPI Local UI against the same DB on port 7847.
#   5. Runs the Playwright spec which screenshots
#      /traces/<failing_id> and /traces/<fixed_id> on the Output tab.
#
# Mirrors scripts/capture-redaction-toggle-screenshots.sh in shape.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT=7847
TMP_ROOT="$(mktemp -d)"
LOCAL_DB="$TMP_ROOT/local.db"
# Point both the trace exporter and the eval scoring at this DB so
# the capture / fix runs and the UI all read/write the same file.
export FASTAIAGENT_LOCAL_DB="$LOCAL_DB"
export FASTAIAGENT_TRACE_DB_PATH="$LOCAL_DB"
trap 'rm -rf "$TMP_ROOT"' EXIT

if ! [ -f .env ]; then
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "▸ OPENAI_API_KEY not set in env. Run via 'zsh -lc' or export it."
    exit 1
  fi
fi

# Kill any stale uvicorn on this port.
if lsof -ti ":$PORT" >/dev/null 2>&1; then
  echo "▸ killing stale process(es) on port $PORT"
  lsof -ti ":$PORT" | xargs kill -9 2>/dev/null || true
  sleep 0.3
fi

echo "▸ running capture.py (buggy run) and fix.py (fork + tool override + rerun)"
cd examples/regression-from-trace
rm -rf .fastaiagent-demo
python capture.py
python fix.py
cd "$REPO_ROOT"

if ! [ -f "$LOCAL_DB" ]; then
  echo "▸ expected local.db at $LOCAL_DB but it doesn't exist"
  exit 1
fi

FAILING_TRACE_ID="$(cat examples/regression-from-trace/.fastaiagent-demo/regression-from-trace/last_trace_id.txt)"
FIXED_TRACE_ID="$(python -c "
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
rows = db.execute(
    'SELECT trace_id FROM spans WHERE name LIKE \"agent.%\" AND trace_id != ? ORDER BY end_time DESC LIMIT 1',
    (sys.argv[2],),
).fetchall()
print(rows[0][0] if rows else '')
" "$LOCAL_DB" "$FAILING_TRACE_ID")"

if [ -z "$FIXED_TRACE_ID" ]; then
  echo "▸ couldn't locate the fixed rerun trace in $LOCAL_DB"
  exit 1
fi
echo "▸ failing trace: $FAILING_TRACE_ID"
echo "▸ fixed trace:   $FIXED_TRACE_ID"

echo "▸ booting UI on port $PORT (no-auth) against $LOCAL_DB"
SERVER_SCRIPT="$TMP_ROOT/serve.py"
cat > "$SERVER_SCRIPT" <<'PY'
import sys
import uvicorn
from fastaiagent.ui.server import build_app

db_path, port = sys.argv[1], int(sys.argv[2])
app = build_app(db_path=db_path, no_auth=True)
uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
PY
python "$SERVER_SCRIPT" "$LOCAL_DB" "$PORT" &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap 'cleanup; rm -rf "$TMP_ROOT"' EXIT

echo "▸ waiting for server"
for _ in $(seq 1 100); do
  if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/api/auth/status"; then
    echo "  server up"
    break
  fi
  sleep 0.2
done

echo "▸ running Playwright regression-from-trace spec"
cd ui-frontend
PLAYWRIGHT_BASE_URL="http://127.0.0.1:$PORT" \
  FAILING_TRACE_ID="$FAILING_TRACE_ID" \
  FIXED_TRACE_ID="$FIXED_TRACE_ID" \
  npx playwright test tests/regression-from-trace.spec.ts
cd "$REPO_ROOT"

echo "▸ done — docs/ui/screenshots/0_3-regression-from-trace-{failing,fixed}.png updated"
