# Environment variables

Every setting below can be provided as an environment variable. Where a
`connect(...)` keyword or CLI flag also exists, the explicit argument wins over
the environment. Booleans accept `1`/`true`/`yes`/`on` (and their negatives);
paths are expanded (`~` works).

!!! tip "Security-relevant variables"
    The 🔒 rows change what leaves your machine or how the SDK is exposed. See
    [Security Posture](../security.md) for the full rationale.

## Connection & platform

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_API_KEY` | — | API key used by `connect()` / the CLI when not passed explicitly. |
| `FASTAIAGENT_TARGET` | `https://app.fastaiagent.net` | Platform base URL. |
| `FASTAIAGENT_PROJECT` | — | Project override for the trace-export payload. |
| `FASTAIAGENT_CONSOLE_URL` | = target | Console origin for console deep links (split-origin dev). |
| `FASTAIAGENT_GOVERNANCE_FAIL_MODE` | `open` | 🔒 `closed` makes governed tool calls refuse when the policy can't be confirmed; `open` preserves fail-open. |

## Egress controls (what leaves the machine)

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_TRACE_ENABLED` | `1` | Master switch. `0` disables tracing entirely (no local capture, no export). |
| `FASTAIAGENT_TRACE_PAYLOADS` | `1` | 🔒 `0` keeps prompts/outputs/tool-args **local only** — they stay in `local.db` (UI/Replay still work) but are stripped before spans reach the plane or any `add_exporter` target. Local capture is always full fidelity. |
| `FASTAIAGENT_EXPORT_EVALS` | `1` | 🔒 `0` stops Agent-CI verdicts (metadata) from being pushed to the plane. |
| `FASTAIAGENT_EXPORT_CHECKPOINTS` | `1` | 🔒 `0` stops checkpoint **state** replication to the plane. Local durability (SQLite/Postgres) is untouched — same-machine resume still works; you lose cross-machine/runner resume, plane disaster-recovery, and console state visibility. |

## Storage & paths

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_LOCAL_DB` | `.fastaiagent/local.db` | Single SQLite file for traces, checkpoints, evals, prompts, KB. |
| `FASTAIAGENT_TRACE_DB_PATH` | = local db | *(deprecated)* separate trace DB path. |
| `FASTAIAGENT_CHECKPOINT_DB_PATH` | = local db | *(deprecated)* separate checkpoint DB path. |
| `FASTAIAGENT_PROMPT_DIR` | — | *(deprecated)* prompt directory. |
| `FASTAIAGENT_CACHE_DIR` | `.fastaiagent/cache/` | Cache directory. |
| `FASTAIAGENT_KB_DIR` | `~/.fastaiagent/kb` | Knowledge-base collections root (read by the UI). |
| `FASTAIAGENT_DB_KEEP_PERMS` | unset | 🔒 By default `local.db`/dir are tightened to owner-only (`0600`/`0700`) on open, removing group/other access. Set to `1` if you deliberately share the DB with a group. |

## Local UI

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_UI_ENABLED` | `false` | Enable the bundled UI via config. |
| `FASTAIAGENT_UI_HOST` | `127.0.0.1` | UI bind host. Non-loopback needs `--insecure-bind`. |
| `FASTAIAGENT_UI_PORT` | `7842` | UI bind port. |
| `FASTAIAGENT_UI_ALLOWED_HOSTS` | loopback only | 🔒 Comma-separated extra `Host` values to accept (anti DNS-rebinding). Add your proxy hostname when fronting the UI. |
| `FASTAIAGENT_UI_TRUST_PROXY` | unset | 🔒 `1` makes the login/LLM rate limiters trust the first `X-Forwarded-For` hop. Set **only** behind a real reverse proxy; otherwise clients could spoof it to dodge throttling. |

## Network, TLS & SSRF

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_LLM_VERIFY` | on | 🔒 `false` disables TLS verification for LLM traffic **that didn't specify `verify=` explicitly** (an explicit `verify=True` is never downgraded); a path value is used as a CA bundle. Prefer the CA-bundle form over disabling. |
| `FASTAIAGENT_ALLOW_PRIVATE_NETWORKS` | unset | 🔒 `1` lets the SSRF-guarded fetchers (multimodal, `RESTTool`, `MCPTool`) reach private/intranet hosts. Loopback is already allowed for `MCPTool`. |
| `FASTAIAGENT_RUNNER_ALLOW_INSECURE` | unset | 🔒 `1` allows `fastaiagent runner --connect` to a **non-loopback** plane over plaintext `http`. By default a remote plane must be `https`. |

## `fastaiagent agent serve`

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_SERVE_TOKEN` | unset | 🔒 Bearer token required on `/run` and `/run/stream`. Set it for any network-exposed deployment (the default bind is `0.0.0.0` for containers). |

## Diagnostics

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_LOG_LEVEL` | `WARNING` | SDK log level. |
| `FASTAIAGENT_DEFAULT_TIMEOUT` | `120` | Default request timeout (seconds). |
