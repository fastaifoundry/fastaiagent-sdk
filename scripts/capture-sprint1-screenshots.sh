#!/usr/bin/env bash
# Sprint 1 screenshot capture. Boots a Local UI server with the Sprint 1
# fixtures (multimodal trace, durable checkpoints, registered Chain) and
# runs the Sprint 1 Playwright spec, which writes evidence PNGs into
# docs/ui/screenshots/.
#
# Independent of scripts/capture-ui-screenshots.sh — this server has a
# Chain registered with build_app(runners=[...]) so the topology endpoint
# returns data, which the legacy snapshot doesn't cover.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TMP_ROOT="$(mktemp -d)"
TMP_DB="$TMP_ROOT/sprint1.db"
trap 'rm -rf "$TMP_ROOT"' EXIT

echo "▸ seeding base UI snapshot DB at $TMP_DB"
python scripts/seed_ui_snapshot.py "$TMP_DB"

echo "▸ overlaying Sprint 1 fixtures (multimodal trace, checkpoints)"
python scripts/seed_ui_sprint1.py "$TMP_DB"

echo "▸ starting FastAPI on port 7844 (--no-auth, snapshot DB, runners=[chain], project=sprint1-demo)"
# Tag every read with project_id="sprint1-demo" so the breadcrumb shows
# the project name and the leakage tests have something to assert on.
# All seed rows in the snapshot DB also stamp project_id="sprint1-demo".
python scripts/_sprint1_ui_server.py \
    --db "$TMP_DB" --host 127.0.0.1 --port 7844 \
    --project-id sprint1-demo &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "▸ waiting for server to accept connections"
for _ in $(seq 1 50); do
  if curl -fsS -o /dev/null http://127.0.0.1:7844/api/auth/status; then
    break
  fi
  sleep 0.2
done

echo "▸ running Sprint 1 Playwright spec"
cd ui-frontend
PLAYWRIGHT_BASE_URL=http://127.0.0.1:7844 npx playwright test tests/sprint1.spec.ts
cd -

echo "▸ done — docs/ui/screenshots/sprint1-*.png updated"
