#!/usr/bin/env bash
# Spawn an *auth-enabled* FastAPI Local UI server (the existing
# ``capture-ui-screenshots.sh`` uses ``--no-auth`` which bypasses the
# CSRF middleware), run the v1.11.0 security Playwright spec, then
# tear everything down.
#
# Closes the "real Chromium" gap in the v1.11.0 PR for:
#   * M1 — iframe sandbox attribute as the browser sees it
#   * M4 — CSRF double-submit token round-trip via real fetch()
#
# Independent of the docs-screenshot orchestration; this one needs auth
# so the CSRF middleware actually runs.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT=7847
TMP_ROOT="$(mktemp -d)"
TMP_DB="$TMP_ROOT/sec.db"
TMP_AUTH="$TMP_ROOT/auth.json"
trap 'rm -rf "$TMP_ROOT"' EXIT

# Kill any stale uvicorn left over from a previous interrupted run on
# this port — otherwise our new server fails to bind silently and
# Playwright sends requests to the stale process whose tempdir has
# already been cleaned up.
if lsof -ti ":$PORT" >/dev/null 2>&1; then
  echo "▸ killing stale process(es) on port $PORT"
  lsof -ti ":$PORT" | xargs kill -9 2>/dev/null || true
  sleep 0.3
fi

echo "▸ seeding snapshot DB at $TMP_DB"
python scripts/seed_ui_snapshot.py "$TMP_DB"

echo "▸ creating auth.json with fixture credentials"
python -c "
from pathlib import Path
from fastaiagent.ui.auth import create_auth_file
create_auth_file('alice', 'correct-horse-battery-staple', path=Path('$TMP_AUTH'))
"

echo "▸ starting FastAPI on port $PORT (AUTH ENABLED, snapshot DB)"
python -c "
import uvicorn
from pathlib import Path
from fastaiagent.ui.server import build_app
app = build_app(db_path='$TMP_DB', auth_path=Path('$TMP_AUTH'), no_auth=False)
uvicorn.run(app, host='127.0.0.1', port=$PORT, log_level='warning')
" >"$TMP_ROOT/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

echo "▸ waiting for server to accept connections"
ready=0
for _ in $(seq 1 50); do
  if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/api/auth/status" 2>/dev/null; then
    ready=1
    break
  fi
  # If uvicorn already died (e.g. bind failure), abort fast with logs.
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "▸ uvicorn exited before accepting connections; server log:"
    cat "$TMP_ROOT/server.log"
    exit 1
  fi
  sleep 0.2
done
if [ "$ready" != "1" ]; then
  echo "▸ server never accepted connections after 10s; log:"
  cat "$TMP_ROOT/server.log"
  exit 1
fi

echo "▸ running Playwright security spec"
cd ui-frontend
PLAYWRIGHT_BASE_URL=http://127.0.0.1:7847 \
PLAYWRIGHT_FIXTURE_USER=alice \
PLAYWRIGHT_FIXTURE_PASSWORD=correct-horse-battery-staple \
  npx playwright test tests/security-1110.spec.ts --reporter=list
cd -

echo "▸ done"
