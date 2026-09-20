#!/usr/bin/env bash
# Sprint 3 screenshot capture. Boots a Local UI server with Sprint 3
# fixtures (compare-pair traces, two seeded datasets, 20 filter-friendly
# traces) and runs the Sprint 3 Playwright spec, which writes evidence
# PNGs into docs/ui/screenshots/.
#
# Independent of capture-sprint{1,2}-screenshots.sh: Sprint 3 needs
# trace data shaped for the new comparison view + datasets on disk
# for the editor, none of which the earlier sprints set up.
#
# No API key required — every screenshot path runs against seeded data.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TMP_ROOT="$(mktemp -d)"
TMP_FA="$TMP_ROOT/.fastaiagent"
mkdir -p "$TMP_FA"
TMP_DB="$TMP_FA/local.db"
trap 'rm -rf "$TMP_ROOT"' EXIT

echo "▸ seeding Sprint 3 fixtures into $TMP_DB"
python scripts/seed_ui_sprint3.py "$TMP_DB" --project-id sprint3-demo

echo "▸ starting FastAPI on port 7847 (--no-auth, project=sprint3-demo)"
python scripts/_sprint3_ui_server.py \
    --db "$TMP_DB" --host 127.0.0.1 --port 7847 \
    --project-id sprint3-demo &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "▸ waiting for server to accept connections"
for _ in $(seq 1 50); do
  if curl -fsS -o /dev/null http://127.0.0.1:7847/api/auth/status; then
    break
  fi
  sleep 0.2
done

echo "▸ running Sprint 3 Playwright spec"
cd ui-frontend
PLAYWRIGHT_BASE_URL=http://127.0.0.1:7847 \
  SPRINT3_TRACE_A=trace-compare-terse \
  SPRINT3_TRACE_B=trace-compare-verbose \
  npx playwright test tests/sprint3.spec.ts
cd -

echo "▸ done — docs/ui/screenshots/sprint3-*.png updated"
