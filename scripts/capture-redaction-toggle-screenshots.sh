#!/usr/bin/env bash
# Capture screenshots of the v1.14 "Mask secrets" toggle on the trace
# detail page so docs/security.md has visual proof of the redaction
# round-trip.
#
# What this does:
#   1. Seeds a temporary local.db with one trace containing fake
#      secrets (sk-PROD..., 4111-1111-1111-1111).
#   2. Boots the production-built UI server against that DB on port
#      7846, with a RedactionPolicy(mode="both") installed so the
#      toggle has something to redact.
#   3. Runs the Playwright spec at
#      ui-frontend/tests/redaction-toggle.spec.ts which screenshots
#      the off-then-on states.
#   4. Tears the server down.
#
# Mirrors scripts/capture-example-sweep-screenshots.sh in shape and
# scripts/capture-security-1110-playwright.sh in needing extra in-process
# setup (a redaction policy) before serving traffic.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT=7846
TMP_ROOT="$(mktemp -d)"
TMP_DB="$TMP_ROOT/redact.db"
trap 'rm -rf "$TMP_ROOT"' EXIT

# Kill any stale uvicorn left from a previous interrupted run.
if lsof -ti ":$PORT" >/dev/null 2>&1; then
  echo "▸ killing stale process(es) on port $PORT"
  lsof -ti ":$PORT" | xargs kill -9 2>/dev/null || true
  sleep 0.3
fi

echo "▸ seeding $TMP_DB with a leaky trace"
export TMP_DB
TMP_DB="$TMP_DB" python - <<'PY'
import json
import os
from datetime import datetime, timezone

from fastaiagent.ui.db import init_local_db

db_path = os.environ["TMP_DB"]
db = init_local_db(db_path)
now = datetime.now(tz=timezone.utc).isoformat()
attrs = {
    "agent.name": "demo-leaky-bot",
    "fastaiagent.runner.type": "agent",
    "agent.input": "Give me a key",
    "agent.output": (
        "Here is your secret key: sk-PROD12345678901234567890ABCDEFGH "
        "and card 4111-1111-1111-1111"
    ),
    "gen_ai.response.content": (
        "Here is your secret key: sk-PROD12345678901234567890ABCDEFGH "
        "and card 4111-1111-1111-1111"
    ),
    "gen_ai.request.model": "gpt-4o-mini",
    "fastaiagent.cost.total_usd": 0.0012,
}
db.execute(
    """INSERT INTO spans
       (span_id, trace_id, parent_span_id, name, start_time, end_time,
        status, attributes, events)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    (
        "demo-redact-span",
        "demo-redact-trace",
        None,
        "agent.demo-leaky-bot",
        now,
        now,
        "OK",
        json.dumps(attrs),
        "[]",
    ),
)
db.close()
PY

echo "▸ booting UI on port $PORT (no-auth, mode='both' redaction policy)"
SERVER_SCRIPT="$TMP_ROOT/serve.py"
cat > "$SERVER_SCRIPT" <<'PY'
import sys
import uvicorn
from fastaiagent.trace import RedactionPolicy, set_redaction_policy
from fastaiagent.ui.server import build_app

db_path, port = sys.argv[1], int(sys.argv[2])
set_redaction_policy(
    RedactionPolicy(
        patterns=(r"sk-[A-Za-z0-9]{20,}", r"\b\d{4}-\d{4}-\d{4}-\d{4}\b"),
        replacement="[REDACTED]",
        mode="both",
    )
)
app = build_app(db_path=db_path, no_auth=True)
uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
PY
python "$SERVER_SCRIPT" "$TMP_DB" "$PORT" &
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

echo "▸ running Playwright redaction-toggle spec"
cd ui-frontend
PLAYWRIGHT_BASE_URL="http://127.0.0.1:$PORT" \
  npx playwright test tests/redaction-toggle.spec.ts
cd -

echo "▸ done — docs/ui/screenshots/0_2-redaction-toggle-{off,on}.png updated"
