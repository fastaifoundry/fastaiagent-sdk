"""End-to-end verification of project scoping.

Scenario (mirrors the multi-project Postgres setup, simulated with SQLite):
  * Two project directories share one DB file.
  * Each project's process resolves its own project_id from
    ``./.fastaiagent/config.toml`` (created on first SDK call, lazily).
  * Each calls ``Agent.run()`` so the SDK writes a real trace through
    the real OTel pipeline → save_span → stamps project_id on the row.
  * Two real FastAPI servers boot, one scoped to each project.
  * We hit every read endpoint and assert project-beta cannot see
    project-alpha's data.

No mocking: real SQLite, real FastAPI, real ``Agent.run()``, real
project_id resolution, real ``init_local_db`` migration. The only
stand-in is the LLM client — a hand-rolled fake from the repo's
``tests/conftest.py`` (not a mock library) that returns a canned
response. Running real OpenAI here just costs money and isn't on the
scoping contract's path.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, "/Users/upendrabhandari/fastaiagent-sdk")

from fastapi.testclient import TestClient

from fastaiagent import Agent
from fastaiagent._internal import project as project_module
from fastaiagent.ui.db import init_local_db
from fastaiagent.ui.server import build_app

# Reuse the repo's hand-rolled LLM fake (not a mock library — it's a real
# Python subclass of LLMClient that returns canned responses).
sys.path.insert(0, "/Users/upendrabhandari/fastaiagent-sdk")
from tests.conftest import MockLLMClient

# ---------------------------------------------------------------------------
# Scenario setup
# ---------------------------------------------------------------------------

ROOT = Path(tempfile.mkdtemp(prefix="sprint1-verify-"))
SHARED_DB = ROOT / "shared.db"
PROJECT_ALPHA_DIR = ROOT / "project-alpha"
PROJECT_BETA_DIR = ROOT / "project-beta"
PROJECT_ALPHA_DIR.mkdir()
PROJECT_BETA_DIR.mkdir()

print(f"▸ scenario root: {ROOT}")
print(f"  shared DB:    {SHARED_DB}")
print(f"  project-alpha: {PROJECT_ALPHA_DIR}")
print(f"  project-beta:  {PROJECT_BETA_DIR}")

# Both projects point at the same DB so we can prove isolation works
# even when they share storage.
os.environ["FASTAIAGENT_LOCAL_DB"] = str(SHARED_DB)

# Pre-create the schema with v4 migration applied.
init_local_db(SHARED_DB).close()


def run_agent_in_project(project_dir: Path, agent_name: str) -> str:
    """Chdir to a project directory, run a real Agent, return its trace_id.

    Resets the ``ProjectConfig`` singleton so each project resolves
    independently (matching what happens when two separate processes
    each have their own cwd).
    """
    os.chdir(project_dir)
    project_module.reset_for_testing()

    # First SDK call → ProjectConfig.load_or_create() →
    # creates ./.fastaiagent/config.toml + .gitignore
    agent = Agent(name=agent_name, llm=MockLLMClient())
    result = agent.run("hello")

    print(f"  [{project_dir.name}] agent={agent_name} trace_id={result.trace_id[:12]}…")
    config_toml = project_dir / ".fastaiagent" / "config.toml"
    assert config_toml.exists(), f"missing {config_toml}"
    print(f"    .fastaiagent/config.toml: {config_toml.read_text().strip()}")
    assert (project_dir / ".fastaiagent" / ".gitignore").exists()
    return result.trace_id


print("\n▸ Step 1: run agents in two project dirs (writes real spans to shared DB)")
alpha_trace = run_agent_in_project(PROJECT_ALPHA_DIR, "alpha-agent")
beta_trace = run_agent_in_project(PROJECT_BETA_DIR, "beta-agent")

print("\n▸ Step 2: probe the DB directly to confirm rows carry project_id")
from fastaiagent._internal.storage import SQLiteHelper

with SQLiteHelper(str(SHARED_DB)) as db:
    rows = db.fetchall(
        "SELECT trace_id, name, project_id FROM spans WHERE parent_span_id IS NULL"
    )
    for r in rows:
        print(
            f"  trace_id={r['trace_id'][:12]}…  name={r['name']}  "
            f"project_id={r['project_id']!r}"
        )
    pids = {r["project_id"] for r in rows}
    assert pids == {"project-alpha", "project-beta"}, (
        f"unexpected project_ids: {pids}"
    )
    print(f"  ✓ both expected project_ids present and distinct")

print("\n▸ Step 3: boot two scoped UI servers against the shared DB")

alpha_app = build_app(
    db_path=str(SHARED_DB), no_auth=True, project_id="project-alpha"
)
beta_app = build_app(
    db_path=str(SHARED_DB), no_auth=True, project_id="project-beta"
)
alpha = TestClient(alpha_app)
beta = TestClient(beta_app)


def expect_only(client: TestClient, label: str, expected: str, forbidden: str) -> None:
    """Hit every read endpoint with both clients and assert isolation."""
    endpoints = [
        f"/api/traces",
        f"/api/traces/{expected}",
        f"/api/traces/{expected}/spans",
        f"/api/traces/{expected}/scores",
        f"/api/traces/threads",
        f"/api/agents",
        f"/api/agents/{label}-agent",
        f"/api/agents/{label}-agent/tools",
        f"/api/workflows",
        f"/api/analytics?hours=24",
        f"/api/analytics/costs?group_by=agent&period=1d",
        f"/api/overview",
        f"/api/auth/status",
    ]
    print(f"\n  ── {label} client (project_id={label!r}) ──")
    for path in endpoints:
        r = client.get(path)
        status = r.status_code
        body = r.text
        leaked = forbidden in body
        leak_marker = "  ✗ LEAK" if leaked else "  ✓"
        print(f"    {leak_marker}  {status}  {path}")
        assert not leaked, (
            f"LEAK: {label} client saw {forbidden!r} on {path}\n  body: {body[:300]}"
        )


print("\n▸ Step 4: per-project endpoint sweep")

# project-alpha must NOT see beta_trace OR 'beta-agent'
expect_only(
    alpha,
    label="project-alpha",
    expected=alpha_trace,
    forbidden=beta_trace,
)
expect_only(
    alpha,
    label="project-alpha",
    expected=alpha_trace,
    forbidden="beta-agent",
)

# project-beta must NOT see alpha_trace OR 'alpha-agent'
expect_only(
    beta,
    label="project-beta",
    expected=beta_trace,
    forbidden=alpha_trace,
)
expect_only(
    beta,
    label="project-beta",
    expected=beta_trace,
    forbidden="alpha-agent",
)

print("\n▸ Step 5: cross-project per-id lookups must 404")
# alpha tries to fetch beta's trace
r = alpha.get(f"/api/traces/{beta_trace}")
print(f"  alpha → /traces/{beta_trace[:12]}…  {r.status_code}")
assert r.status_code == 404, f"expected 404, got {r.status_code}"

# beta tries to fetch alpha's agent detail
r = beta.get(f"/api/agents/alpha-agent")
print(f"  beta  → /agents/alpha-agent          {r.status_code}")
assert r.status_code == 404

# Even though both trace_ids exist in the DB, each project gets only its own.
print("\n▸ Step 6: confirm /api/auth/status reports the right project")
print(f"  alpha auth.status: {alpha.get('/api/auth/status').json()}")
print(f"  beta  auth.status: {beta.get('/api/auth/status').json()}")
assert alpha.get("/api/auth/status").json()["project_id"] == "project-alpha"
assert beta.get("/api/auth/status").json()["project_id"] == "project-beta"

print("\n▸ Cleanup")
shutil.rmtree(ROOT)
print("\n✅ ALL VERIFICATIONS PASSED — project scoping holds end-to-end.")
