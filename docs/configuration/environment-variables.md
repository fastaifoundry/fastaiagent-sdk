# Environment variables

Every setting below can be provided as an environment variable. Where a
`connect(...)` keyword or CLI flag also exists, the explicit argument wins over
the environment.

**Booleans** accept `1`, `true`, `yes`, `on` and `0`, `false`, `no`, `off` —
case-insensitive, surrounding whitespace ignored. **An empty value counts as
unset** (a docker-compose `FOO:` or a Kubernetes `value: ""` means "I didn't set
this"), so it resolves to the documented default.

!!! warning "An unrecognised value on a 🔒 row fails closed"
    A typo is never silently favourable. `FASTAIAGENT_TRACE_PAYLOADS=ture`
    resolves to **off** (payloads stay local), and
    `FASTAIAGENT_ALLOW_PRIVATE_NETWORKS=ture` resolves to **not granted**. Every
    unparsed value logs a `WARNING` naming the variable, the value you set, and
    the accepted spellings, so it is visible rather than merely safe.

**Paths** expand `~` and `$VARS`. A relative path is resolved against the
process's working directory, which is why `~/…` is the right form for anything
two processes must share.

!!! tip "Security-relevant variables"
    The 🔒 rows change what leaves your machine or how the SDK is exposed. See
    [Security Posture](../security.md) for the full rationale.

`fastaiagent.config` exposes the same settings in code — see
[SDK configuration](sdk-config.md).

## Connection & platform

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_API_KEY` | — | API key used by `connect()` / the CLI when not passed explicitly. |
| `FASTAIAGENT_TARGET` | `https://app.fastaiagent.net` | Platform base URL. |
| `FASTAIAGENT_PROJECT` | — | Project override for the trace-export payload. |
| `FASTAIAGENT_CONSOLE_URL` | = target | Console origin for console deep links (split-origin dev). |
| `FASTAIAGENT_GOVERNANCE_FAIL_MODE` | `open` | 🔒 `closed` makes governed tool calls refuse when the policy can't be confirmed; `open` preserves fail-open. **Not a boolean**: only the literal `closed` hardens the gate, so a typo can never silently fail-*close* a production gate. An unrecognised value logs a warning and stays `open`. |

## Egress controls (what leaves the machine)

`connect()` logs the resolved posture of this whole section once, at `INFO`, so
you can see how your values were read.

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_TRACE_ENABLED` | `1` | 🔒 Master switch. Off disables tracing entirely — **no local capture** (no rows in `local.db`, no attachment bytes, no foreign-span capture) and therefore no export. |
| `FASTAIAGENT_TRACE_PAYLOADS` | `1` | 🔒 Off keeps prompts/outputs/tool-args **local only** — they stay in `local.db` (UI/Replay still work) but are stripped before spans reach the plane or any `add_exporter` target. Local capture is always full fidelity. |
| `FASTAIAGENT_EXPORT_EVALS` | `1` | 🔒 Off stops Agent-CI verdicts (metadata) from being pushed to the plane. |
| `FASTAIAGENT_EXPORT_CHECKPOINTS` | `1` | 🔒 Off stops checkpoint **state** replication to the plane. Local durability (SQLite/Postgres) is untouched — same-machine resume still works; you lose cross-machine/runner resume, plane disaster-recovery, and console state visibility. |
| `FASTAIAGENT_RESTORE_FROM_PLANE` | `1` | 🔒 Off stops a resume from pulling a run's state back from the plane when this machine has none. Keep it on for cross-machine resume and disaster recovery; turn it **off** to honour an erasure request, so a run whose local checkpoints were deliberately deleted is not resurrected. Deployment-wide by design — there is no per-call argument. |

## Storage & paths

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_LOCAL_DB` | `.fastaiagent/local.db` | Single SQLite file for traces, checkpoints, evals, prompts, KB. |
| `FASTAIAGENT_TRACE_DB_PATH` | = local db | *(deprecated)* separate trace DB path. |
| `FASTAIAGENT_CHECKPOINT_DB_PATH` | = local db | *(deprecated)* separate checkpoint DB path. |
| `FASTAIAGENT_PROMPT_DIR` | — | *(deprecated)* prompt directory. |
| `FASTAIAGENT_KB_DIR` | `.fastaiagent/kb` | Knowledge-base collections root (read by the UI's KB browser and the agent detail view). |
| `FASTAIAGENT_MODEL_CATALOG` | beside `local.db` | Path to the local UI's model-catalogue override file (JSON). |
| `FASTAIAGENT_DB_KEEP_PERMS` | `0` | 🔒 By default `local.db`/dir are tightened to owner-only (`0600`/`0700`) on open, removing group/other access. Turn on if you deliberately share the DB with a group. |
| `FASTAIAGENT_CACHE_DIR` | `.fastaiagent/cache/` | Sets `config.cache_dir`. **Read by nothing in the SDK today** — the library keeps no disk cache of its own. Kept because it is a public field; see [SDK configuration](sdk-config.md). |

## Local UI

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_UI_ENABLED` | `0` | Enable the bundled UI via config. |
| `FASTAIAGENT_UI_HOST` | `127.0.0.1` | UI bind host, used when `fastaiagent ui` is started without `--host`. Non-loopback needs `--insecure-bind`. |
| `FASTAIAGENT_UI_PORT` | `7842` | UI bind port, used when `fastaiagent ui` is started without `--port`. |
| `FASTAIAGENT_UI_ALLOWED_HOSTS` | loopback only | 🔒 Comma-separated extra `Host` values to accept (anti DNS-rebinding). Add your proxy hostname when fronting the UI. |
| `FASTAIAGENT_UI_TRUST_PROXY` | `0` | 🔒 On makes the login/LLM rate limiters trust the first `X-Forwarded-For` hop. Set **only** behind a real reverse proxy; otherwise clients could spoof it to dodge throttling. |

## Multimodal

These set the process-wide defaults for every `LLMClient`; an explicit keyword
argument on a client always wins. See [SDK configuration](sdk-config.md).

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_PDF_MODE` | `auto` | `auto` \| `text` \| `vision` \| `native` — how PDFs are sent to the model. |
| `FASTAIAGENT_MAX_PDF_PAGES` | `20` | Page cap for vision-mode rendering. |
| `FASTAIAGENT_MAX_IMAGE_SIZE_MB` | provider's own cap | Per-image ceiling. Unset means each provider's own limit applies (Anthropic 5 MB, OpenAI 20 MB, …); setting it overrides them all. |
| `FASTAIAGENT_TRACE_FULL_IMAGES` | `0` | Store original image/PDF bytes in `local.db` alongside the 256-px thumbnail, so Replay can fork with the exact payload. Local only — see [Security Posture](../security.md#attachment-bytes). |

## Network, TLS & SSRF

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_LLM_VERIFY` | on | 🔒 **Tri-state, not a boolean**: a false value (`0`/`false`/`no`/`off`) disables TLS verification for LLM traffic **that didn't specify `verify=` explicitly** (an explicit `verify=True` is never downgraded); a true value forces it on; anything else is used as a CA-bundle path (`~` expands). Prefer the CA-bundle form over disabling. |
| `FASTAIAGENT_ALLOW_PRIVATE_NETWORKS` | `0` | 🔒 On lets the SSRF-guarded fetchers (multimodal, `RESTTool`, `MCPTool`) reach private/intranet hosts. Loopback is already allowed for `MCPTool`. |
| `FASTAIAGENT_RUNNER_ALLOW_INSECURE` | `0` | 🔒 On allows `fastaiagent runner --connect` to a **non-loopback** plane over plaintext `http`. By default a remote plane must be `https`. |

## `fastaiagent agent serve`

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_SERVE_TOKEN` | unset | 🔒 Bearer token required on `/run` and `/run/stream`. Set it for any network-exposed deployment (the default bind is `0.0.0.0` for containers). |

## Diagnostics

| Variable | Default | Purpose |
|---|---|---|
| `FASTAIAGENT_LOG_LEVEL` | `WARNING` | Sets `config.log_level`. **The SDK installs no logging handler** — it uses standard `logging` and inherits your application's configuration, so this field is informational. Configure verbosity with `logging.getLogger("fastaiagent").setLevel(...)`. |
| `FASTAIAGENT_DEFAULT_TIMEOUT` | `120` | Sets `config.default_timeout`. **Read by nothing in the SDK today** — every HTTP timeout is passed explicitly at its call site (`connect()` uses 10s, the outboxes 10s, `LLMClient` its own). Kept because it is a public field. |
