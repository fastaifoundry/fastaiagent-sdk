#!/usr/bin/env bash
# Seed a real foreign-instrumentor span, boot the UI against it, and run the
# foreign-OTel Playwright spec (asserts rich rendering + saves screenshots).
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-7846}"
TMP_DB="$(mktemp -t foreign-otel-XXXX.db)"

echo "▸ seeding foreign OpenInference span into $TMP_DB"
TRACE_ID="$(python scripts/seed_foreign_otel.py "$TMP_DB")"
echo "  trace_id=$TRACE_ID"

echo "▸ starting UI server on 127.0.0.1:$PORT (no-auth, unscoped)"
python scripts/_otel_ui_server.py --db "$TMP_DB" --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  rm -f "$TMP_DB"
}
trap cleanup EXIT

echo "▸ waiting for server"
for _ in $(seq 1 50); do
  if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/api/auth/status"; then break; fi
  sleep 0.2
done

echo "▸ running Playwright spec"
cd ui-frontend
PLAYWRIGHT_BASE_URL="http://127.0.0.1:$PORT" FOREIGN_TRACE_ID="$TRACE_ID" \
  npx playwright test tests/foreign-otel-live.spec.ts
