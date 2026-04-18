# Deploy with Docker — Cloud Run / Fly / Render / Railway / ECS

Once you have the [FastAPI server](fastapi.md), a single `Dockerfile` gives you a portable artifact that runs on every container platform. This page shows the Dockerfile and the one-command deploy for each of the big four.

## Dockerfile

Put this next to your `server.py`:

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# System deps (faiss needs libstdc++, pymupdf needs libgl1 if PDFs).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libstdc++6 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY server.py .

# Non-root user (platforms like Cloud Run prefer this)
RUN useradd -m app && chown -R app:app /app
USER app

# Cloud Run / Fly / Render all pass $PORT; default to 8000 locally.
ENV PORT=8000
EXPOSE 8000

CMD exec uvicorn server:app --host 0.0.0.0 --port ${PORT}
```

`requirements.txt`:

```
fastaiagent[kb]>=0.6.0
fastapi>=0.115
uvicorn>=0.32
```

Adjust extras to match what your agent uses — `[openai]`, `[anthropic]`, `[qdrant]`, `[chroma]`, `[mcp-server]`, etc.

## Build and test locally

```bash
docker build -t my-agent .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... my-agent
curl localhost:8000/health
```

## Deploy — Google Cloud Run

Cloud Run scales to zero, charges per request, and handles HTTPS + load balancing for you. Ideal first production target.

```bash
# Build and push to Google Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev
docker tag my-agent us-central1-docker.pkg.dev/$PROJECT/agents/my-agent:v1
docker push us-central1-docker.pkg.dev/$PROJECT/agents/my-agent:v1

# Deploy
gcloud run deploy my-agent \
  --image us-central1-docker.pkg.dev/$PROJECT/agents/my-agent:v1 \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 20 \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest
```

**Notes:**
- `--timeout 300` — agent tool loops can be slow; default is 300s max on Cloud Run.
- `--min-instances 1` if you want zero cold starts (costs ~$10/mo for the always-on instance).
- Secrets live in Google Secret Manager; don't bake API keys into the image.
- Add `--no-allow-unauthenticated` + an API gateway / IAP if the endpoint is private.

## Deploy — Fly.io

Fly is per-second-billed and great for bursty traffic:

```bash
fly launch              # generates fly.toml, asks a few questions
fly secrets set OPENAI_API_KEY=sk-...
fly deploy
```

A reasonable `fly.toml`:

```toml
app = "my-agent"
primary_region = "iad"

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true   # scale to zero
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  memory = "1gb"
  cpu_kind = "shared"
  cpus = 1
```

## Deploy — Render

Render is the simplest path. Push your repo, point Render at it:

1. In the Render dashboard: **New → Web Service → Connect Git repo**.
2. Runtime: **Docker**.
3. Start command: leave blank (uses `CMD` from the Dockerfile).
4. Environment: add `OPENAI_API_KEY` (marked secret).
5. Instance type: **Starter ($7/mo)** for low traffic, **Standard ($25/mo)** for steady load.

Render builds the image on their infrastructure and hands you an HTTPS URL. No CLI needed.

## Deploy — Railway

Same experience as Render; CLI-driven:

```bash
railway init
railway up
railway variables set OPENAI_API_KEY=sk-...
```

Railway detects the Dockerfile automatically and hands back a public URL.

## Deploy — AWS ECS / Fargate

For teams already on AWS. No one-liner; use Terraform or CDK. The key template fragment:

```hcl
resource "aws_ecs_task_definition" "agent" {
  family = "my-agent"
  network_mode = "awsvpc"
  cpu    = 512
  memory = 1024
  requires_compatibilities = ["FARGATE"]

  container_definitions = jsonencode([{
    name  = "agent"
    image = "your-ecr-repo/my-agent:v1"
    portMappings = [{ containerPort = 8000 }]
    secrets = [
      { name = "OPENAI_API_KEY", valueFrom = "arn:aws:secretsmanager:..." }
    ]
  }])
}
```

Front with an ALB; use ECS Service Autoscaling for scale-out.

## Production checklist

| | Handled by platform | You do |
|---|---|---|
| HTTPS / TLS | Cloud Run, Fly, Render, Railway, ALB all terminate HTTPS | — |
| Secrets | Use platform secret manager | Don't bake into image |
| Logs | Platform streams stdout | Log a structured JSON line per request if you want searchability |
| Metrics | Platform has basic CPU/memory | Add OTel export to the fastaiagent Platform for per-run traces |
| Autoscaling | Cloud Run, Fly, Render scale automatically | Set sane max to avoid runaway cost |
| Cold start | Cloud Run ~1s; Fly / Render / Railway warm instances | `--min-instances 1` removes cold start at a monthly cost |

## Observability in production

The fastaiagent Platform already handles observability. Wire it up at server startup:

```python
# In server.py, inside build_agent() or at module load
import fastaiagent as fa
import os

if os.environ.get("FASTAIAGENT_PLATFORM_URL"):
    fa.connect(
        os.environ["FASTAIAGENT_PLATFORM_URL"],
        api_key=os.environ["FASTAIAGENT_API_KEY"],
    )
```

Set those two env vars in your platform's secret manager and every production run's trace lands in your dashboard next to your dev runs.

---

## Next

- [FastAPI server code](fastapi.md) — the app this Dockerfile runs
- [Modal](modal.md) — no-container alternative
- [Streaming](../streaming/index.md) — server-sent events details
