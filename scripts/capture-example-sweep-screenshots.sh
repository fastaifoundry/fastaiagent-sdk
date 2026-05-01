#!/usr/bin/env bash
# Capture screenshots of every Local UI surface against the
# repo's live ``.fastaiagent/local.db`` — populated by running the
# full example sweep first. Use to verify that every example
# produced visible data through the UI.
#
# Boots the server scoped to ``project_id=fastaiagent-sdk`` (the
# value the lifecycle resolved when each example wrote its rows).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DB_PATH="$REPO_ROOT/.fastaiagent/local.db"
if [ ! -f "$DB_PATH" ]; then
  echo "▸ no local.db at $DB_PATH — run examples first to populate it"
  exit 1
fi

echo "▸ starting FastAPI on port 7845 (no-auth, repo's local.db, project=fastaiagent-sdk)"
python -c "
import uvicorn
from fastaiagent.ui.server import build_app
app = build_app(db_path='$DB_PATH', no_auth=True, project_id='fastaiagent-sdk')
uvicorn.run(app, host='127.0.0.1', port=7845, log_level='warning')
" &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "▸ waiting for server"
for _ in $(seq 1 50); do
  if curl -fsS -o /dev/null http://127.0.0.1:7845/api/auth/status; then
    break
  fi
  sleep 0.2
done

echo "▸ running Playwright spec"
cd ui-frontend
PLAYWRIGHT_BASE_URL=http://127.0.0.1:7845 npx playwright test tests/example-sweep.spec.ts
cd -

echo "▸ done — docs/ui/screenshots/examples-*.png updated"
