#!/usr/bin/env bash
# Seed a snapshot DB, start the Local UI server, capture docs screenshots,
# then tear everything down. Used to keep docs/ui/screenshots/ in sync.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TMP_DB="$(mktemp -d)/ui-snapshot.db"
trap 'rm -rf "$(dirname "$TMP_DB")"' EXIT

echo "▸ seeding snapshot DB at $TMP_DB"
python scripts/seed_ui_snapshot.py "$TMP_DB"

echo "▸ starting FastAPI on port 7843 (--no-auth, snapshot DB)"
python -c "
import sys, uvicorn
from fastaiagent.ui.server import build_app
app = build_app(db_path='$TMP_DB', no_auth=True)
uvicorn.run(app, host='127.0.0.1', port=7843, log_level='warning')
" &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  rm -rf "$(dirname "$TMP_DB")"
}
trap cleanup EXIT

echo "▸ waiting for server to accept connections"
for _ in $(seq 1 50); do
  if curl -fsS -o /dev/null http://127.0.0.1:7843/api/auth/status; then
    break
  fi
  sleep 0.2
done

echo "▸ running Playwright screenshot spec"
cd ui-frontend
PLAYWRIGHT_BASE_URL=http://127.0.0.1:7843 npx playwright test tests/screenshots.spec.ts
cd -

echo "▸ done — docs/ui/screenshots/*.png updated"
