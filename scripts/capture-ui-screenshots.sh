#!/usr/bin/env bash
# Seed a snapshot DB, start the Local UI server, capture docs screenshots,
# then tear everything down. Used to keep docs/ui/screenshots/ in sync.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TMP_ROOT="$(mktemp -d)"
TMP_DB="$TMP_ROOT/ui-snapshot.db"
TMP_KB_ROOT="$TMP_ROOT/kb"
trap 'rm -rf "$TMP_ROOT"' EXIT

echo "▸ seeding snapshot DB at $TMP_DB"
python scripts/seed_ui_snapshot.py "$TMP_DB"

echo "▸ seeding LocalKB at $TMP_KB_ROOT (+retrieval spans)"
python scripts/seed_ui_kb.py "$TMP_KB_ROOT" --db "$TMP_DB"

echo "▸ starting FastAPI on port 7843 (--no-auth, snapshot DB, KB root=$TMP_KB_ROOT)"
FASTAIAGENT_KB_DIR="$TMP_KB_ROOT" python -c "
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
