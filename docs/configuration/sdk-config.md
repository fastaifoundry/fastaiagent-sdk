# SDK configuration (`fastaiagent.config`)

`fastaiagent.config` is the process-wide settings object. It is the same thing
[environment variables](environment-variables.md) set — the environment is read
once, on first access, and `fastaiagent.config` is what the rest of the library
reads from then on.

```python
import fastaiagent as fa

fa.config.pdf_mode = "vision"
fa.config.trace_full_images = True

print(fa.config.local_db_path)   # '.fastaiagent/local.db'
```

It is a mutable [pydantic](https://docs.pydantic.dev) model, so an assignment
sticks for the life of the process and is picked up by everything constructed
afterwards. Set it **before** you build the objects that read it (an `LLMClient`
resolves its multimodal settings at construction), and before the run you want
traced.

!!! note "One object, resolved once"
    `fa.config` is a singleton. `fa.config is fa.config` and there is no
    per-agent override — per-object settings are constructor arguments on the
    object itself (`LLMClient(pdf_mode=...)`), and those always win.

## Fields

### Storage & paths

| Field | Env | Default | What reads it |
|---|---|---|---|
| `local_db_path` | `FASTAIAGENT_LOCAL_DB` | `.fastaiagent/local.db` | Traces, checkpoints, evals, prompts, KB — everything local. |
| `trace_db_path` | `FASTAIAGENT_TRACE_DB_PATH` | `None` → `local_db_path` | *Deprecated.* Read via `config.resolved_trace_db_path`. |
| `checkpoint_db_path` | `FASTAIAGENT_CHECKPOINT_DB_PATH` | `None` → `local_db_path` | *Deprecated.* Read via `config.resolved_checkpoint_db_path`. |
| `prompt_dir` | `FASTAIAGENT_PROMPT_DIR` | `None` | *Deprecated.* Prompt file loader. |
| `kb_dir` | `FASTAIAGENT_KB_DIR` | `.fastaiagent/kb` | The local UI's KB browser and the agent detail view. |
| `cache_dir` | `FASTAIAGENT_CACHE_DIR` | `.fastaiagent/cache/` | **Nothing.** See [Inert fields](#inert-fields). |

Paths expand `~` and `$VARS`, both from the environment and when handed
directly to a CLI flag such as `fastaiagent ui --db ~/work/local.db`. Prefer an
absolute or `~`-rooted path whenever two processes must agree on the same store:
a relative path resolves against each process's working directory, so a run
started in one directory and resumed in another will not find its checkpoints.

### Tracing

| Field | Env | Default | What reads it |
|---|---|---|---|
| `trace_enabled` | `FASTAIAGENT_TRACE_ENABLED` | `True` | `trace.otel.get_tracer_provider` — off means a no-op tracer, so nothing is captured **or** exported. |
| `trace_full_images` | `FASTAIAGENT_TRACE_FULL_IMAGES` | `False` | `trace.attachments` — stores original image/PDF bytes next to the thumbnail. |

### Multimodal

These are the process-wide defaults for every `LLMClient`. A keyword argument on
a client always wins over them.

| Field | Env | Default | What reads it |
|---|---|---|---|
| `pdf_mode` | `FASTAIAGENT_PDF_MODE` | `"auto"` | `LLMClient` → `multimodal.format`. |
| `max_pdf_pages` | `FASTAIAGENT_MAX_PDF_PAGES` | `20` | Vision-mode page cap. |
| `max_image_size_mb` | `FASTAIAGENT_MAX_IMAGE_SIZE_MB` | `None` | Per-image ceiling. |

!!! warning "`max_image_size_mb=None` means *the provider's* cap, not "no cap""
    Each provider has its own per-image limit — Anthropic 5 MB, OpenAI 20 MB —
    and the default `None` selects it. Setting a number overrides **every**
    provider, in both directions: `fa.config.max_image_size_mb = 20.0` would
    raise Anthropic's effective ceiling fourfold and get your requests rejected
    by the API rather than resized locally. Lower it freely; raise it only for a
    provider you know accepts it.

### Local UI

| Field | Env | Default | What reads it |
|---|---|---|---|
| `ui_enabled` | `FASTAIAGENT_UI_ENABLED` | `False` | Opt-in flag for embedding the bundled UI. |
| `ui_host` | `FASTAIAGENT_UI_HOST` | `127.0.0.1` | `fastaiagent ui` when started without `--host`. |
| `ui_port` | `FASTAIAGENT_UI_PORT` | `7842` | `fastaiagent ui` when started without `--port`. |

### Inert fields

Two fields are parsed and exposed but read by nothing in the library. They are
documented here rather than quietly removed, because they are public and have
shipped environment variables:

| Field | Why it does nothing |
|---|---|
| `cache_dir` | The SDK keeps no disk cache of its own. Prompt/KB/trace storage all live in `local.db`. |
| `log_level` | The SDK installs no logging handler — it uses standard `logging` and inherits your application's configuration. Use `logging.getLogger("fastaiagent").setLevel(...)`. |
| `default_timeout` | Every HTTP timeout is passed explicitly at its call site (`connect()` 10s, the durable outboxes 10s, `LLMClient` its own retry/timeout settings). |

## Reading the resolved values

```python
import fastaiagent as fa

print(fa.config.model_dump())
```

`connect()` additionally logs the resolved egress posture once, at `INFO`:

```
fastaiagent egress posture: tracing=on payloads=OFF traces=on checkpoints=OFF evals=on
```

That line is the quickest way to confirm the environment was read the way you
meant it — the switches it summarises are described in
[Security Posture](../security.md).

## Resetting (tests)

```python
from fastaiagent._internal.config import reset_config

reset_config()   # drop the cached singleton; the next access re-reads the env
```
