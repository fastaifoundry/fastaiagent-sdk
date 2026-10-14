# Shared shell plumbing for the 1.89.0 capture scripts. Source it, then:
#
#     cap_setup <port>            # temp dir, local.db path, clean env, cleanup trap
#     "$PY" seed.py …             # seed $LOCAL_DB with real SDK calls (cwd = $TMP_ROOT)
#     cap_start_ui [--agent …]    # `fastaiagent ui start --db $LOCAL_DB --no-auth` from $TMP_ROOT
#     "$PY" scripts/capture_x.py "$BASE_URL" …
#
# Everything runs from a temp working directory: the repo's .fastaiagent/ holds
# legacy stores that the UI would otherwise import into the screenshot database.
# The server is stopped and the temp directory removed on exit.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
FA="${FASTAIAGENT_BIN:-$REPO_ROOT/.venv/bin/fastaiagent}"
SERVER_PID=""

cap_cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ -n "${KEEP_TMP:-}" ]; then
    echo "▸ kept $TMP_ROOT"
  else
    rm -rf "$TMP_ROOT"
  fi
}

cap_setup() {
  PORT="${PORT:-$1}"
  if lsof -ti ":$PORT" >/dev/null 2>&1; then
    echo "port $PORT is busy — set PORT=<free port>" >&2
    exit 1
  fi
  # The physical path (`pwd -P`): macOS temp dirs live under the /var ->
  # /private/var symlink, and `ui start --db` compares a resolved path with an
  # unresolved one when it decides whether a Playground trace is in the UI's
  # store (see the 1.89.0 capture report). Resolving here keeps the shots about
  # the feature, not that defect.
  TMP_ROOT="$(cd "$(mktemp -d)" && pwd -P)"
  LOCAL_DB="$TMP_ROOT/.fastaiagent/local.db"
  mkdir -p "$TMP_ROOT/.fastaiagent"
  BASE_URL="http://127.0.0.1:$PORT"
  trap cap_cleanup EXIT
  # Never talk to a plane, and keep every store in this one file.
  unset FASTAIAGENT_API_KEY FASTAIAGENT_TARGET FASTAIAGENT_TRACE_DB_PATH \
    FASTAIAGENT_CHECKPOINT_DB_PATH FASTAIAGENT_UI_ENABLED FASTAIAGENT_REPLAY_TOOL_POLICY \
    FASTAIAGENT_EXPORT_EXPERIMENTS FASTAIAGENT_UI_PROVIDER_ERRORS
  export FASTAIAGENT_LOCAL_DB="$LOCAL_DB"
  export FASTAIAGENT_KB_DIR="$TMP_ROOT/.fastaiagent/kb"
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  cd "$TMP_ROOT"
}

# Offline scripts: no provider key reaches the seed, the UI or the browser, so
# nothing can call a real model and the shots don't depend on the shell.
cap_offline() {
  unset OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY GOOGLE_API_KEY MISTRAL_API_KEY \
    GROQ_API_KEY TOGETHER_API_KEY DEEPSEEK_API_KEY XAI_API_KEY OPENROUTER_API_KEY \
    AZURE_OPENAI_API_KEY COHERE_API_KEY
}

# The Playground lists a provider as selectable only when its key is set. A
# dummy key makes openai/custom selectable; every offline run also sets an
# Endpoint (the localhost SSE stub) and a Token, and the server refuses an
# endpoint without a token, so this value is never sent anywhere.
cap_dummy_openai_key() {
  export OPENAI_API_KEY="sk-capture-dummy-0000000000000000000000000000"
}

cap_start_ui() {
  echo "▸ fastaiagent ui start --db $LOCAL_DB --no-auth --port $PORT $*"
  (cd "$TMP_ROOT" && exec "$FA" ui start --db "$LOCAL_DB" --no-auth --no-open --port "$PORT" "$@") \
    > "$TMP_ROOT/ui.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 150); do
    curl -fsS -o /dev/null "$BASE_URL/api/auth/status" 2>/dev/null && return 0
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    sleep 0.2
  done
  echo "the UI did not start:" >&2
  cat "$TMP_ROOT/ui.log" >&2
  exit 1
}
