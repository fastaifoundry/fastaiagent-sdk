# Platform API Layer (Internals)

This document explains how the SDK communicates with the FastAIAgent Platform — the shared HTTP client, the connection lifecycle, and how each feature (PromptRegistry, Traces, Eval, Replay) uses it. It's written for contributors who need to add new platform-facing features, debug connectivity issues, or understand the caching and error-handling behavior.

For the user-facing platform guide, see [docs/platform/index.md](../platform/index.md).
For prompt user docs, see [docs/prompts/index.md](../prompts/index.md).
For the tracing internals (span lifecycle, SQLite storage, OTLP export), see [tracing-architecture.md](tracing-architecture.md).

---

## Architecture Overview

```
fa.connect(api_key, target)
    │
    ▼
_Connection singleton (client.py)
    │  api_key, target, domain_id, project_id, scopes
    │
    ▼
PlatformAPI (httpx client)  ←── _platform/api.py
    │
    ├── GET  /public/v1/auth/check           ← fa.connect() auth
    │
    ├── POST /public/v1/prompts              ← PromptRegistry.publish()
    ├── GET  /public/v1/prompts/{slug}       ← PromptRegistry.get(source="platform")
    │
    ├── POST /public/v1/traces/ingest        ← PlatformSpanExporter (background, batched)
    ├── GET  /public/v1/traces/{trace_id}    ← Replay.from_platform()
    │
    ├── POST /public/v1/eval/datasets        ← Dataset.publish()
    ├── GET  /public/v1/eval/datasets/{name} ← Dataset.from_platform()
    ├── POST /public/v1/eval/runs            ← EvalResults.publish()
    │
    └── Headers on every request:
            X-API-Key: {api_key}
            Content-Type: application/json
            User-Agent: fastaiagent-sdk/{version}
```

Every platform-facing feature in the SDK goes through the same `PlatformAPI` HTTP client. The only exception is `PlatformSpanExporter`, which uses its own `httpx.Client` instance for trace ingest (because it runs in a `BatchSpanProcessor` background thread and needs its own connection lifecycle).

---

## Connection Lifecycle

### The `_Connection` Singleton

**File:** `fastaiagent/client.py` (lines 13–38)

```python
class _Connection:
    def __init__(self):
        self.api_key: str | None = None
        self.target: str = "https://app.fastaiagent.net"
        self.project: str | None = None
        self.domain_id: str | None = None
        self.project_id: str | None = None
        self.scopes: list[str] = []
        self._platform_processor: Any = None   # BatchSpanProcessor for traces

    @property
    def is_connected(self) -> bool:
        return self.api_key is not None

    @property
    def headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self.api_key or "",
            "Content-Type": "application/json",
            "User-Agent": f"fastaiagent-sdk/{__version__}",
        }

_connection = _Connection()   # Process-wide singleton
```

Created once at module load. Every platform feature imports `_connection` directly and checks `_connection.is_connected`.

### What `fa.connect()` Does (Step by Step)

**File:** `fastaiagent/client.py` (lines 61–136)

```
fa.connect(api_key="fa_k_...", target="localhost:8001", project="my-project")
    │
    ├── 1. Store credentials on _connection
    │       api_key = "fa_k_..."
    │       target = _normalize_target("localhost:8001")  → "http://localhost:8001"
    │       project = "my-project"
    │
    ├── 2. Auth check: GET {target}/public/v1/auth/check
    │       Headers: X-API-Key: fa_k_...
    │       │
    │       ├── 200 → store domain_id, project_id, scopes from response
    │       │         log "Connected to platform: domain=... project=... scopes=..."
    │       │
    │       ├── 401 → clear api_key, raise PlatformAuthError("Invalid API key")
    │       │
    │       ├── 403 → clear api_key, raise PlatformAuthError
    │       │
    │       └── ConnectError (unreachable) → log warning, keep connection stored
    │                "Connection stored — traces will export when platform is reachable."
    │                (Optimistic connect: platform doesn't have to be up right now)
    │
    └── 3. Register platform trace exporter
            PlatformSpanExporter() → BatchSpanProcessor(exporter)
            → get_tracer_provider().add_span_processor(processor)
            → stored on _connection._platform_processor for disconnect()
```

**Target URL normalization** (`_normalize_target()`, lines 41–58):
- `"localhost:8001"` → `"http://localhost:8001"` (auto-prepends `http://` for localhost)
- `"app.fastaiagent.net"` → `"https://app.fastaiagent.net"` (auto-prepends `https://` for public hosts)
- `"http://localhost:8001"` → unchanged
- Strips trailing slashes

### What `fa.disconnect()` Does

**File:** `fastaiagent/client.py` (lines 139–152)

```
fa.disconnect()
    │
    ├── _platform_processor.force_flush(timeout_millis=5000)
    │       Drains any pending spans from the BatchSpanProcessor
    │
    ├── _platform_processor.shutdown()
    │       Stops the background thread
    │
    └── Clears connection state
            api_key = None, project = None, etc.
```

After disconnect, traces go to local SQLite only. `is_connected` returns `False`. Any platform-facing call (publish, get, etc.) either raises `PlatformNotConnectedError` or silently falls back to local.

---

## The PlatformAPI HTTP Client

**File:** `fastaiagent/_platform/api.py`

### Construction

```python
class PlatformAPI:
    def __init__(self, api_key: str, base_url: str, timeout: int = 30):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
```

Not a singleton — a new instance is created by `get_platform_api()` on every call. This is intentional: it reads the current `_connection.api_key` and `_connection.target` at call time, so if the user disconnects and reconnects with different credentials, the next API call uses the new ones.

### `get_platform_api()` Factory

**File:** `fastaiagent/_platform/api.py` (lines 148–159)

```python
def get_platform_api() -> PlatformAPI:
    from fastaiagent.client import _connection
    if not _connection.is_connected:
        raise PlatformNotConnectedError("Not connected. Call fa.connect() first.")
    return PlatformAPI(
        api_key=_connection.api_key,
        base_url=_connection.target,
    )
```

Every platform-facing method calls this before making an HTTP request.

### Methods

| Method | Signature | What it does |
|--------|-----------|-------------|
| `get(path, params)` | Sync GET → `dict` | `httpx.Client().get(url, params=params, headers=...)` → parse JSON |
| `post(path, data)` | Sync POST → `dict` | `httpx.Client().post(url, json=data, headers=...)` → parse JSON |
| `aget(path, params)` | Async GET → `dict` | `httpx.AsyncClient().get(...)` |
| `apost(path, data)` | Async POST → `dict` | `httpx.AsyncClient().post(...)` |

All four methods call `_handle_response(response)` which handles errors uniformly:

### Error Handling

**File:** `fastaiagent/_platform/api.py` (lines 43–81)

| HTTP Status | Exception Raised | Detection Logic |
|-------------|-----------------|-----------------|
| 401 | `PlatformAuthError` | Always |
| 403 | `PlatformTierLimitError` | If response body contains "tier" |
| 403 | `PlatformAuthError` | Otherwise |
| 404 | `PlatformNotFoundError` | Always |
| 429 | `PlatformRateLimitError` | Includes `Retry-After` header value in the exception |
| 500+ | `PlatformConnectionError` | Any server error |
| 200–299 | (success) | Returns `response.json()` |

This error mapping is shared by every feature. A contributor adding a new platform endpoint gets this error handling for free by using `api.get()` / `api.post()`.

---

## Feature: Prompt Registry (Platform Path)

**File:** `fastaiagent/prompt/registry.py`

### `publish(slug, content, variables)` — Push to Platform

```
PromptRegistry.publish("support-v1", "You are a {{role}}", ["role"])
    │
    ├── _is_connected()? No → raise PlatformNotConnectedError
    │
    ├── get_platform_api()
    │
    └── api.post("/public/v1/prompts", {
            "slug": "support-v1",
            "content": "You are a {{role}}",
            "variables": ["role"]
        })
```

No return value on success. Raises on any HTTP error via `_handle_response()`.

### `get(slug, version, source)` — Fetch from Platform or Local

Three modes depending on `source` parameter:

**`source="platform"` (explicit platform):**
```
get("support-v1", source="platform")
    │
    ├── _fetch_from_platform("support-v1", version=None)
    │       │
    │       ├── Cache hit? → return cached Prompt immediately
    │       │
    │       ├── api.get("/public/v1/prompts/support-v1")
    │       │       Response: {"slug", "content", "variables", "version", "metadata"}
    │       │
    │       ├── Parse into Prompt(name, template, variables, version, metadata)
    │       │
    │       ├── Cache with TTL: _platform_cache[("support-v1", None)] = (prompt, now + 300s)
    │       │
    │       └── Return Prompt
    │
    ├── Found → return
    └── Not found → raise PromptNotFoundError
```

**`source="auto"` (default — platform-first with local fallback):**
```
get("support-v1", source="auto")
    │
    ├── is_connected()?
    │       ├── Yes → _fetch_from_platform("support-v1")
    │       │       ├── Found → return
    │       │       └── Not found or error → fall through silently
    │       └── No → skip platform
    │
    └── _fetch_from_local("support-v1")
            └── Reads .prompts/support-v1.json from disk
```

**`source="local"` (explicit local):**
```
get("support-v1", source="local")
    │
    └── _fetch_from_local("support-v1")
            └── Reads .prompts/support-v1.json from disk
```

### Platform Cache

**File:** `fastaiagent/prompt/registry.py` (lines 142–178)

```python
_DEFAULT_CACHE_TTL = 300  # 5 minutes

# Cache structure: {(slug, version): (Prompt, expires_at)}
self._platform_cache: dict[tuple[str, int | None], tuple[Prompt, float]] = {}
```

**Cache behavior:**
- Key: `(slug, version)` — so `get("x", version=1)` and `get("x", version=2)` are cached separately
- TTL: 300 seconds from the time of fetch
- On hit: returns the cached `Prompt` object directly (0ms, no HTTP)
- On expiry: deletes the entry, makes a fresh HTTP call
- On `refresh(slug)`: removes ALL entries where the slug matches (all versions)

**Cache is per-instance**, not global. If you create two `PromptRegistry()` instances, they have independent caches. In practice most users create one.

### `refresh(slug)` — Cache Invalidation

**File:** `fastaiagent/prompt/registry.py` (lines 83–87)

```python
def refresh(self, slug):
    keys_to_remove = [k for k in self._platform_cache if k[0] == slug]
    for k in keys_to_remove:
        del self._platform_cache[k]
```

Use this when you know the platform-side prompt has changed (e.g., after a colleague publishes a new version) and you want the next `get()` to bypass the cache.

---

## Feature: Prompt Registry (Local Path)

### Local Storage: `YAMLStorage`

**File:** `fastaiagent/prompt/storage.py`

Despite the name, `YAMLStorage` uses JSON files (not YAML) under `.prompts/`:

```
.prompts/
├── support.json                     ← prompt "support" with all versions
│   {
│       "name": "support",
│       "versions": [
│           {"name": "support", "template": "...", "variables": [...], "version": 1, "metadata": {}},
│           {"name": "support", "template": "...", "variables": [...], "version": 2, "metadata": {}}
│       ],
│       "latest_version": 2,
│       "aliases": {"production": 1, "staging": 2}
│   }
│
├── _fragment_tone.json              ← fragment "tone"
│   {"name": "tone", "content": "Be professional.", "version": 1}
│
└── _fragment_safety.json            ← fragment "safety"
    {"name": "safety", "content": "Never disclose internal info.", "version": 1}
```

### `load_prompt(name, version, alias)`

**File:** `fastaiagent/prompt/storage.py` (lines 33–56)

```
load_prompt("support")           → latest version (version=latest_version)
load_prompt("support", version=1) → specific version
load_prompt("support", alias="production") → version mapped by alias
```

Resolution priority:
1. If `alias` provided → look up version number from `data["aliases"][alias]`
2. If `version` provided → find matching entry in `data["versions"]`
3. If neither → return latest (version = `data["latest_version"]`)

### Versioning

Each `register()` call auto-increments the version number:

```python
reg.register(name="support", template="v1 prompt")  # → version 1
reg.register(name="support", template="v2 prompt")  # → version 2
reg.register(name="support", template="v3 prompt")  # → version 3

prompt = reg.load("support")          # → version 3 (latest)
prompt = reg.load("support", version=1)  # → version 1
```

Unless `version` is explicitly passed:

```python
reg.register(name="support", template="...", version=10)  # → version 10
```

---

## Feature: Fragment Composition

### Registration

**File:** `fastaiagent/prompt/registry.py` (lines 116–121)

```python
def register_fragment(self, name, content):
    fragment = Fragment(name=name, content=content)
    self._storage.save_fragment(fragment)
    self._fragments[name] = fragment
```

Stored both in memory (`self._fragments` dict) and on disk (`.prompts/_fragment_{name}.json`).

### Resolution: `{{@fragment_name}}` Syntax

**File:** `fastaiagent/prompt/registry.py` (lines 185–202)

When `load()` is called, the template's `{{@fragment_name}}` references are resolved before the `Prompt` object is returned:

```python
def _resolve_fragments(self, template):
    pattern = r"\{\{@(\w+)\}\}"

    def replacer(match):
        frag_name = match.group(1)
        # 1. Check in-memory cache
        if frag_name in self._fragments:
            return self._fragments[frag_name].content
        # 2. Load from disk
        try:
            frag = self._storage.load_fragment(frag_name)
            self._fragments[frag_name] = frag
            return frag.content
        except Exception:
            return match.group(0)  # Leave unresolved

    return re.sub(pattern, replacer, template)
```

**Important:** Fragment resolution happens at `load()` time, NOT at `format()` time. The resolved template is what gets cached and returned. So `format()` only handles `{{variable}}` substitution.

### Fragment vs Variable Syntax

| Syntax | Resolved when | By what |
|--------|--------------|---------|
| `{{@tone}}` | At `load()` time | `_resolve_fragments()` — replaced with fragment content |
| `{{name}}` | At `format()` time | `Prompt.format(name="World")` — replaced with kwarg value |

---

## Feature: Trace Platform Push (via PlatformSpanExporter)

**File:** `fastaiagent/trace/platform_export.py`

This is the ONLY platform feature that does NOT use `PlatformAPI`. It has its own `httpx.Client` because it runs in a `BatchSpanProcessor` background thread.

```
Span ends → on_end()
    │
    ├── LocalStorageProcessor → SQLite (sync, immediate)
    │
    └── BatchSpanProcessor (background thread)
            │
            └── PlatformSpanExporter.export(spans)
                    │
                    ├── Check _connection.is_connected → skip if false
                    │
                    ├── Convert spans to dicts (same shape as SQLite rows)
                    │
                    └── POST {target}/public/v1/traces/ingest
                            {
                                "project": "...",
                                "spans": [
                                    {"span_id", "trace_id", "name", "attributes": {...}, ...}
                                ]
                            }
```

See [tracing-architecture.md](tracing-architecture.md) for the full span lifecycle.

### Trace Fetch: `Replay.from_platform(trace_id)`

**File:** `fastaiagent/trace/replay.py`

The platform API returns a different span schema than local SQLite. The `from_platform()` method maps between them.

```
Replay.from_platform("abc123")
    │
    ├── Check _connection.is_connected → raise PlatformNotConnectedError if not
    │
    ├── get_platform_api()
    │
    ├── api.get("/public/v1/traces/abc123")
    │       Platform response:
    │       {
    │           "id": "abc123...",         ← trace-level, maps to trace_id
    │           "source": "sdk",
    │           "status": "completed",
    │           "total_tokens": 98,
    │           "total_duration_ms": 2923,
    │           "spans": [
    │               {
    │                   "id": "44aea...",  ← maps to span_id (NOT span_id)
    │                   "span_type": "sdk",
    │                   "name": "agent.support-bot",
    │                   "status": "unset",
    │                   "input": { ...all span attributes... },
    │                   "output": { ...may have additional attrs... },
    │                   "start_time": "...",
    │                   "end_time": "...",
    │                   "metadata": {}
    │               }
    │           ]
    │       }
    │
    ├── Map each platform span to SDK SpanData:
    │       span["id"]                    → SpanData.span_id
    │       trace-level data["id"]        → SpanData.trace_id
    │       None                          → SpanData.parent_span_id (not in platform response)
    │       span["input"] + span["output"]→ SpanData.attributes  (merged into one dict)
    │       span["name"]                  → SpanData.name
    │       span["start_time/end_time"]   → SpanData.start_time/end_time
    │
    └── Construct TraceData(trace_id, name, spans=[...]) → Replay(trace_data)
```

**Key schema differences** (platform vs local SQLite):

| Field | Platform response | SDK `SpanData` | Mapping |
|-------|------------------|----------------|---------|
| Span identifier | `"id"` | `span_id` | `s["id"]` → `span_id` |
| Trace identifier | On trace envelope `data["id"]` | On each span `trace_id` | Propagated from envelope |
| Parent span | Not provided | `parent_span_id` | Set to `None` |
| Attributes | Split: `"input"` + `"output"` dicts | Single `attributes` dict | Merged: `input.update(output)` |
| Extra fields | `span_type`, `metadata` | Not in model | Ignored |

This mapping means `Replay.from_platform()` traces and `Replay.load()` traces produce the same `TraceData` / `SpanData` shape downstream — `fork_at()`, `rerun()`, and `compare()` work identically regardless of the source.

---

## Feature: Eval Platform Push

### `Dataset.publish(name)` and `Dataset.from_platform(name)`

**File:** `fastaiagent/eval/dataset.py`

```
Dataset.publish("golden-set")
    │
    ├── Check _connection.is_connected
    ├── get_platform_api()
    └── api.post("/public/v1/eval/datasets", {"name": "golden-set", "items": [...]})

Dataset.from_platform("golden-set")
    │
    ├── Check _connection.is_connected
    ├── get_platform_api()
    └── api.get("/public/v1/eval/datasets/golden-set")
            → Dataset(items=data["items"])
```

### `EvalResults.publish(run_name)`

**File:** `fastaiagent/eval/results.py`

```
results.publish("v1-golden")
    │
    ├── Check _connection.is_connected
    ├── get_platform_api()
    └── api.post("/public/v1/eval/runs", {
            "run_name": "v1-golden",
            "scores": {
                "contains_keyword": [
                    {"score": 1.0, "passed": true, "reason": "..."},
                    {"score": 0.0, "passed": false, "reason": "..."}
                ]
            }
        })
```

---

## Complete Platform Endpoint Table

| Feature | Direction | Method | Endpoint | Payload / Params |
|---------|-----------|--------|----------|-----------------|
| Auth | SDK → platform | GET | `/public/v1/auth/check` | — |
| Prompt publish | SDK → platform | POST | `/public/v1/prompts` | `{slug, content, variables}` |
| Prompt fetch | Platform → SDK | GET | `/public/v1/prompts/{slug}` | `?version=N` (optional) |
| Trace ingest | SDK → platform | POST | `/public/v1/traces/ingest` | `{project, spans: [...]}` |
| Trace fetch | Platform → SDK | GET | `/public/v1/traces/{trace_id}` | — |
| Dataset publish | SDK → platform | POST | `/public/v1/eval/datasets` | `{name, items: [...]}` |
| Dataset fetch | Platform → SDK | GET | `/public/v1/eval/datasets/{name}` | — |
| Eval publish | SDK → platform | POST | `/public/v1/eval/runs` | `{run_name, scores: {...}}` |

All endpoints are under the `/public/v1/` prefix. All require `X-API-Key` header. All return JSON.

---

## How Each Feature Checks the Connection

Every platform-facing method follows the same pattern. If you're adding a new platform feature, copy this:

```python
def my_platform_method(self, ...):
    from fastaiagent._internal.errors import PlatformNotConnectedError
    from fastaiagent._platform.api import get_platform_api
    from fastaiagent.client import _connection

    if not _connection.is_connected:
        raise PlatformNotConnectedError(
            "Not connected to platform. Call fa.connect() first."
        )
    api = get_platform_api()
    data = api.post("/public/v1/my-endpoint", {"key": "value"})
    # or: data = api.get("/public/v1/my-endpoint/{id}")
    return data
```

The connection check + `get_platform_api()` + `api.get/post` pattern is used identically by:
- `PromptRegistry.publish()` and `_fetch_from_platform()`
- `Dataset.publish()` and `Dataset.from_platform()`
- `EvalResults.publish()`
- `Replay.from_platform()`

---

## Graceful Degradation Patterns

Different features degrade differently when the platform is unavailable:

| Feature | Behavior when not connected | Behavior when connected but platform is down |
|---------|---------------------------|----------------------------------------------|
| **Traces** | Go to local SQLite only. No error. | `BatchSpanProcessor` queues spans; `PlatformSpanExporter.export()` returns SUCCESS silently on failure. Traces still land in SQLite. |
| **PromptRegistry.get(source="auto")** | Falls back to local storage. No error. | Catches the exception, returns None from `_fetch_from_platform()`, falls back to local. |
| **PromptRegistry.get(source="platform")** | Raises `PlatformNotConnectedError` | Raises `PlatformConnectionError` (500+) or `PlatformNotFoundError` (404) |
| **PromptRegistry.publish()** | Raises `PlatformNotConnectedError` | Raises the appropriate platform error |
| **Dataset / EvalResults publish** | Raises `PlatformNotConnectedError` | Raises the appropriate platform error |
| **Replay.from_platform()** | Raises `PlatformNotConnectedError` | Raises the appropriate platform error |

The key design insight: **read paths degrade gracefully** (auto source detection falls back to local), but **write paths fail loudly** (you asked to publish something to the platform and it's not reachable — that's an error worth knowing about).

---

## Common Contributor Patterns

### Adding a new platform-facing feature

1. Add your endpoint to the platform endpoint table above
2. In your feature module, follow the connection-check pattern (see "How Each Feature Checks the Connection" above)
3. Use `api.get()` / `api.post()` from `get_platform_api()` — you get error handling for free
4. If your feature is a read path, consider the `source="auto"` pattern (platform-first with local fallback)
5. If your feature benefits from caching, copy the `_platform_cache` TTL pattern from PromptRegistry

### Testing platform features in the e2e gate

Platform-dependent gate steps are gated by `require_platform()` in `tests/e2e/conftest.py`. This skips the step when `E2E_SKIP_PLATFORM=1` is set (CI mode). Locally, with the platform running, all steps run.

See `tests/e2e/test_gate_prompt_registry.py` for the canonical pattern: `TestPromptRegistryLocalGate` (always runs) + `TestPromptRegistryPlatformGate` (platform-gated).

---

## Files Reference

| File | What it does |
|------|-------------|
| `fastaiagent/client.py` | `_Connection` singleton, `connect()`, `disconnect()`, `_normalize_target()` |
| `fastaiagent/_platform/api.py` | `PlatformAPI` HTTP client, `get_platform_api()` factory, error handling |
| `fastaiagent/prompt/registry.py` | `PromptRegistry` — publish, get, load, fragments, TTL cache |
| `fastaiagent/prompt/storage.py` | `YAMLStorage` — local JSON file storage for prompts and fragments |
| `fastaiagent/prompt/prompt.py` | `Prompt` model — template, variables, format() |
| `fastaiagent/prompt/fragment.py` | `Fragment` model — name, content |
| `fastaiagent/eval/dataset.py` | `Dataset.publish()`, `Dataset.from_platform()` |
| `fastaiagent/eval/results.py` | `EvalResults.publish()` |
| `fastaiagent/trace/platform_export.py` | `PlatformSpanExporter` — background trace push |
| `fastaiagent/trace/replay.py` | `Replay.from_platform()` — pull traces from platform |
| `fastaiagent/_internal/errors.py` | All `Platform*Error` exception classes |
