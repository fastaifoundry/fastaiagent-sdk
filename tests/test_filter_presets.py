"""Saved filter preset tests (Sprint 3).

Real SQLite + real FastAPI TestClient. Project scoping is exercised
against two distinct ``build_app(project_id=...)`` instances pointed
at the same DB — same-DB-multi-project is the production case the
scoping is meant to handle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


@pytest.fixture
def app_db(temp_dir: Path):
    db_path = temp_dir / "local.db"
    db = init_local_db(db_path)
    db.close()
    app = build_app(db_path=str(db_path), no_auth=True)
    return app, db_path


@pytest.fixture
def client(app_db):
    app, _ = app_db
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. POST then GET round-trip
# ---------------------------------------------------------------------------


class TestCreateAndList:
    def test_post_then_list_returns_preset(self, client: TestClient) -> None:
        body = {
            "name": "Errors this week",
            "filters": {"status": "ERROR", "since": "2026-04-26T00:00:00Z"},
        }
        r = client.post("/api/filter-presets", json=body)
        assert r.status_code == 201, r.text
        created = r.json()
        assert created["name"] == "Errors this week"
        assert created["filters"]["status"] == "ERROR"
        assert "id" in created and len(created["id"]) > 0

        listing = client.get("/api/filter-presets").json()
        assert any(p["id"] == created["id"] for p in listing)

    def test_invalid_name_returns_400(self, client: TestClient) -> None:
        r = client.post(
            "/api/filter-presets",
            json={"name": "", "filters": {}},
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# 2. PATCH renames + replaces filters
# ---------------------------------------------------------------------------


class TestPatch:
    def test_rename_only(self, client: TestClient) -> None:
        created = client.post(
            "/api/filter-presets",
            json={"name": "Old", "filters": {"status": "OK"}},
        ).json()
        r = client.patch(
            f"/api/filter-presets/{created['id']}",
            json={"name": "New"},
        )
        assert r.status_code == 200
        assert r.json()["name"] == "New"
        # Filters survived the rename.
        assert r.json()["filters"]["status"] == "OK"

    def test_replace_filters_only(self, client: TestClient) -> None:
        created = client.post(
            "/api/filter-presets",
            json={"name": "P", "filters": {"status": "OK"}},
        ).json()
        r = client.patch(
            f"/api/filter-presets/{created['id']}",
            json={"filters": {"status": "ERROR", "min_cost": 0.05}},
        )
        assert r.status_code == 200
        # Name preserved.
        assert r.json()["name"] == "P"
        assert r.json()["filters"] == {"status": "ERROR", "min_cost": 0.05}


# ---------------------------------------------------------------------------
# 3. DELETE
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_then_get_404(self, client: TestClient) -> None:
        created = client.post(
            "/api/filter-presets",
            json={"name": "trash", "filters": {}},
        ).json()
        d = client.delete(f"/api/filter-presets/{created['id']}")
        assert d.status_code == 204
        # PATCH on the missing id 404s.
        p = client.patch(
            f"/api/filter-presets/{created['id']}",
            json={"name": "ghost"},
        )
        assert p.status_code == 404


# ---------------------------------------------------------------------------
# 4. Project scoping — preset under P1 is invisible to P2.
# ---------------------------------------------------------------------------


class TestProjectScoping:
    def test_preset_isolated_per_project(self, temp_dir: Path) -> None:
        db_path = temp_dir / "local.db"
        db = init_local_db(db_path)
        db.close()

        app1 = build_app(db_path=str(db_path), no_auth=True, project_id="p1")
        app2 = build_app(db_path=str(db_path), no_auth=True, project_id="p2")
        c1 = TestClient(app1)
        c2 = TestClient(app2)

        created = c1.post(
            "/api/filter-presets",
            json={"name": "p1-only", "filters": {"status": "ERROR"}},
        ).json()

        # P1 sees its preset.
        ids1 = [p["id"] for p in c1.get("/api/filter-presets").json()]
        assert created["id"] in ids1

        # P2 doesn't.
        ids2 = [p["id"] for p in c2.get("/api/filter-presets").json()]
        assert created["id"] not in ids2

        # And cross-project DELETE / PATCH are 404, not silent success.
        cross_del = c2.delete(f"/api/filter-presets/{created['id']}")
        assert cross_del.status_code == 404
        cross_patch = c2.patch(
            f"/api/filter-presets/{created['id']}",
            json={"name": "stolen"},
        )
        assert cross_patch.status_code == 404
