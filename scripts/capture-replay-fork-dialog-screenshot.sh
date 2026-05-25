#!/usr/bin/env bash
# Capture the v1.14.1 "Rerun with modifications" dialog state for
# docs/replay/index.md. Boots a fresh UI seeded with one agent trace,
# runs the Playwright spec at
# ui-frontend/tests/replay-fork-dialog.spec.ts, and writes
# docs/ui/screenshots/0_3_audit-rerun-dialog.png.
#
# Mirrors scripts/capture-redaction-toggle-screenshots.sh in shape.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT=7848
TMP_ROOT="$(mktemp -d)"
LOCAL_DB="$TMP_ROOT/local.db"
TRACE_ID="replay-dialog-trace"
trap 'rm -rf "$TMP_ROOT"' EXIT

# Kill any stale uvicorn on the port.
if lsof -ti ":$PORT" >/dev/null 2>&1; then
  echo "▸ killing stale process(es) on port $PORT"
  lsof -ti ":$PORT" | xargs kill -9 2>/dev/null || true
  sleep 0.3
fi

echo "▸ seeding $LOCAL_DB with a single agent trace"
LOCAL_DB="$LOCAL_DB" TRACE_ID="$TRACE_ID" python - <<'PY'
import json
import os
from datetime import datetime, timezone

from fastaiagent.ui.db import init_local_db

db = init_local_db(os.environ["LOCAL_DB"])
now = datetime.now(tz=timezone.utc).isoformat()
attrs = {
    "agent.name": "support-bot",
    "agent.input": "Look up ORD-1",
    "agent.output": "Order ORD-1 was delivered.",
    "agent.system_prompt": "be helpful",
    "agent.config": json.dumps({"max_iterations": 5}),
    "agent.tools": json.dumps([{"name": "lookup_order", "tool_type": "function"}]),
    "agent.guardrails": json.dumps([]),
    "agent.llm.provider": "openai",
    "agent.llm.model": "gpt-4o-mini",
    "agent.llm.config": json.dumps(
        {"provider": "openai", "model": "gpt-4o-mini", "api_key": "k"}
    ),
}
try:
    db.execute(
        """INSERT INTO spans
           (span_id, trace_id, parent_span_id, name, start_time, end_time,
            status, attributes, events)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "root",
            os.environ["TRACE_ID"],
            None,
            "agent.support-bot",
            now,
            now,
            "OK",
            json.dumps(attrs),
            "[]",
        ),
    )
finally:
    db.close()
PY

echo "▸ booting UI on port $PORT (no-auth)"
SERVE="$TMP_ROOT/serve.py"
cat > "$SERVE" <<'PY'
import sys
import uvicorn
from fastaiagent.ui.server import build_app

db_path, port = sys.argv[1], int(sys.argv[2])
app = build_app(db_path=db_path, no_auth=True)
uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
PY
python "$SERVE" "$LOCAL_DB" "$PORT" &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap 'cleanup; rm -rf "$TMP_ROOT"' EXIT

echo "▸ waiting for server (up to ~20s)"
for _ in $(seq 1 100); do
  if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/api/auth/status"; then
    echo "  server up"
    break
  fi
  sleep 0.2
done

echo "▸ running Playwright replay-fork-dialog spec"
cd ui-frontend
PLAYWRIGHT_BASE_URL="http://127.0.0.1:$PORT" \
  REPLAY_FORK_TRACE_ID="$TRACE_ID" \
  npx playwright test tests/replay-fork-dialog.spec.ts
cd -

echo "▸ done — docs/ui/screenshots/0_3_audit-rerun-dialog.png updated"
