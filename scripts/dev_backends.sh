#!/usr/bin/env bash
# Start / stop the Postgres and Redis containers the backend-gated integration
# tests use.
#
#   scripts/dev_backends.sh up     # start both, wait until they answer
#   scripts/dev_backends.sh down   # remove both
#   scripts/dev_backends.sh status
#
# With these running, `tests/integration/conftest.py` detects them and the 39
# tests that otherwise skip on `PG_TEST_DSN` / `REDIS_TEST_URL` execute — no
# environment variables to export. Setting either variable by hand still wins.
#
# Non-default ports (55432 / 56379) so this can never collide with a Postgres
# or Redis you run for your own work; the conftest additionally refuses any
# Postgres whose database is not named `fastaiagent_test`.

set -euo pipefail

PG_NAME="fa-pg-test"
REDIS_NAME="fa-redis-test"
PG_PORT=55432
REDIS_PORT=56379
PG_DSN="postgresql://postgres:test@127.0.0.1:${PG_PORT}/fastaiagent_test"
REDIS_URL="redis://127.0.0.1:${REDIS_PORT}/15"

running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]; }

up() {
  if running "$PG_NAME"; then
    echo "▸ $PG_NAME already running"
  else
    docker rm -f "$PG_NAME" >/dev/null 2>&1 || true
    echo "▸ starting $PG_NAME on 127.0.0.1:$PG_PORT"
    docker run -d --name "$PG_NAME" \
      -e POSTGRES_PASSWORD=test -e POSTGRES_DB=fastaiagent_test \
      -p "127.0.0.1:${PG_PORT}:5432" postgres:16-alpine >/dev/null
  fi

  if running "$REDIS_NAME"; then
    echo "▸ $REDIS_NAME already running"
  else
    docker rm -f "$REDIS_NAME" >/dev/null 2>&1 || true
    echo "▸ starting $REDIS_NAME on 127.0.0.1:$REDIS_PORT"
    docker run -d --name "$REDIS_NAME" \
      -p "127.0.0.1:${REDIS_PORT}:6379" redis:7-alpine >/dev/null
  fi

  echo "▸ waiting for Postgres to accept connections"
  for _ in $(seq 1 60); do
    if docker exec "$PG_NAME" pg_isready -U postgres -q 2>/dev/null; then break; fi
    sleep 1
  done
  docker exec "$PG_NAME" pg_isready -U postgres -q || {
    echo "✗ Postgres did not become ready" >&2
    exit 1
  }

  echo "▸ waiting for Redis to answer PING"
  for _ in $(seq 1 30); do
    if [ "$(docker exec "$REDIS_NAME" redis-cli ping 2>/dev/null)" = "PONG" ]; then break; fi
    sleep 1
  done

  cat <<EOF

✓ backends up. The integration suite finds these on its own:

    pytest tests/integration -q

  To target them explicitly (or from another checkout):
    export PG_TEST_DSN="$PG_DSN"
    export REDIS_TEST_URL="$REDIS_URL"
EOF
}

down() {
  # Exact container names only. Never a broad pattern kill: one has already
  # taken down the local control plane here and made an SDK bug appear.
  for name in "$PG_NAME" "$REDIS_NAME"; do
    if docker inspect "$name" >/dev/null 2>&1; then
      docker rm -f "$name" >/dev/null && echo "▸ removed $name"
    else
      echo "▸ $name not present"
    fi
  done
}

status() {
  for name in "$PG_NAME" "$REDIS_NAME"; do
    if running "$name"; then echo "✓ $name running"; else echo "✗ $name not running"; fi
  done
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  status) status ;;
  *)
    echo "usage: $0 {up|down|status}" >&2
    exit 2
    ;;
esac
