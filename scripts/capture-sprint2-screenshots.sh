#!/usr/bin/env bash
# Sprint 2 screenshot capture. Boots a Local UI server with the Sprint 2
# fixtures (two registered prompts) and runs the Sprint 2 Playwright spec,
# which writes evidence PNGs into docs/ui/screenshots/.
#
# Independent of capture-sprint1-screenshots.sh and capture-ui-screenshots.sh:
# Sprint 2 needs prompts in the registry for the Playground to be testable,
# not chain/swarm/supervisor runners.
#
# The streaming-response screenshot needs OPENAI_API_KEY in your env so the
# Playwright spec can drive a real LLM call. The other shots run without
# any key.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TMP_ROOT="$(mktemp -d)"
TMP_DB="$TMP_ROOT/sprint2.db"
trap 'rm -rf "$TMP_ROOT"' EXIT

echo "▸ seeding base UI snapshot DB at $TMP_DB"
python scripts/seed_ui_snapshot.py "$TMP_DB"

echo "▸ overlaying Sprint 2 fixtures (prompts + guardrail events)"
SEED_OUTPUT="$(python scripts/seed_ui_sprint2.py "$TMP_DB" --project-id sprint2-demo)"
echo "$SEED_OUTPUT"

# Pull the seeded event ids out of the script's stdout so the Playwright
# spec can deep-link to them. The seed script emits a Python repr like:
#   ✓ seeded 3 guardrail events: {'blocked': 'ev-...', 'filtered': '...', 'warned': '...'}
extract() {
  echo "$SEED_OUTPUT" | python -c "
import re, sys
text = sys.stdin.read()
match = re.search(r\"'$1':\s*'([^']+)'\", text)
print(match.group(1) if match else '', end='')
"
}
SPRINT2_BLOCKED_EVENT_ID="$(extract blocked)"
SPRINT2_FILTERED_EVENT_ID="$(extract filtered)"
SPRINT2_WARNED_EVENT_ID="$(extract warned)"
export SPRINT2_BLOCKED_EVENT_ID SPRINT2_FILTERED_EVENT_ID SPRINT2_WARNED_EVENT_ID
echo "▸ event ids — blocked=$SPRINT2_BLOCKED_EVENT_ID filtered=$SPRINT2_FILTERED_EVENT_ID warned=$SPRINT2_WARNED_EVENT_ID"

echo "▸ starting FastAPI on port 7845 (--no-auth, snapshot DB, project=sprint2-demo)"
python scripts/_sprint2_ui_server.py \
    --db "$TMP_DB" --host 127.0.0.1 --port 7845 \
    --project-id sprint2-demo &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "▸ waiting for server to accept connections"
for _ in $(seq 1 50); do
  if curl -fsS -o /dev/null http://127.0.0.1:7845/api/auth/status; then
    break
  fi
  sleep 0.2
done

echo "▸ running Sprint 2 Playwright spec"
cd ui-frontend
PLAYWRIGHT_BASE_URL=http://127.0.0.1:7845 npx playwright test tests/sprint2.spec.ts
cd -

echo "▸ done — docs/ui/screenshots/sprint2-*.png updated"
