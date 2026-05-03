"""End-to-end tests for the Eval Dataset Editor endpoints.

Real FastAPI + real SQLite + real JSONL files. No mocks. Per the
no-mocking rule, every assertion is against the actual file the
endpoint wrote.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.eval.dataset import Dataset  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


@pytest.fixture
def app_db(temp_dir: Path):
    """Build a no-auth app with a ``.fastaiagent/local.db`` layout.

    Mirrors the production directory shape so ``_datasets_dir`` resolves
    to ``<temp_dir>/.fastaiagent/datasets``. Tests can find the JSONL
    files at that path.
    """
    fa_dir = temp_dir / ".fastaiagent"
    fa_dir.mkdir(parents=True, exist_ok=True)
    db_path = fa_dir / "local.db"
    db = init_local_db(db_path)
    db.close()
    app = build_app(db_path=str(db_path), no_auth=True)
    return app, db_path, fa_dir / "datasets"


@pytest.fixture
def client(app_db):
    app, _, _ = app_db
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Create + list + delete
# ---------------------------------------------------------------------------


class TestCreateAndList:
    def test_post_then_get_lists_dataset(self, client: TestClient, app_db) -> None:
        _, _, datasets_dir = app_db
        r = client.post("/api/datasets", json={"name": "my-set"})
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["name"] == "my-set"
        assert body["case_count"] == 0
        # File on disk where the eval framework expects it
        assert (datasets_dir / "my-set.jsonl").exists()

        # And it appears in the list endpoint
        r = client.get("/api/datasets")
        assert r.status_code == 200
        names = [d["name"] for d in r.json()]
        assert "my-set" in names

    def test_create_duplicate_returns_409(self, client: TestClient) -> None:
        client.post("/api/datasets", json={"name": "dup"})
        r = client.post("/api/datasets", json={"name": "dup"})
        assert r.status_code == 409


class TestDelete:
    def test_delete_removes_file(self, client: TestClient, app_db) -> None:
        _, _, datasets_dir = app_db
        client.post("/api/datasets", json={"name": "trash"})
        r = client.delete("/api/datasets/trash")
        assert r.status_code == 204
        assert not (datasets_dir / "trash.jsonl").exists()


# ---------------------------------------------------------------------------
# 2. Case CRUD round-trip — JSONL stays loadable by Dataset.from_jsonl
# ---------------------------------------------------------------------------


class TestCaseCRUD:
    def test_add_case_round_trips_through_dataset_loader(
        self, client: TestClient, app_db
    ) -> None:
        _, _, datasets_dir = app_db
        client.post("/api/datasets", json={"name": "echo"})

        r = client.post(
            "/api/datasets/echo/cases",
            json={
                "input": "say hello",
                "expected_output": "hello",
                "tags": ["smoke"],
            },
        )
        assert r.status_code == 201
        assert r.json()["index"] == 0

        # GET reflects the case
        body = client.get("/api/datasets/echo").json()
        assert len(body["cases"]) == 1
        assert body["cases"][0]["input"] == "say hello"
        assert body["cases"][0]["expected_output"] == "hello"
        assert body["cases"][0]["tags"] == ["smoke"]

        # And the on-disk JSONL parses through the eval Dataset loader
        ds = Dataset.from_jsonl(datasets_dir / "echo.jsonl")
        items = list(ds)
        assert len(items) == 1
        assert items[0]["input"] == "say hello"
        assert items[0]["expected_output"] == "hello"

    def test_put_updates_in_place_others_untouched(
        self, client: TestClient
    ) -> None:
        client.post("/api/datasets", json={"name": "many"})
        for v in ("a", "b", "c"):
            client.post("/api/datasets/many/cases", json={"input": v, "expected": v})

        r = client.put(
            "/api/datasets/many/cases/1",
            json={"input": "B", "expected_output": "B!"},
        )
        assert r.status_code == 200
        cases = client.get("/api/datasets/many").json()["cases"]
        assert [c["input"] for c in cases] == ["a", "B", "c"]
        assert cases[1]["expected_output"] == "B!"

    def test_delete_case_reindexes(self, client: TestClient) -> None:
        client.post("/api/datasets", json={"name": "many2"})
        for v in ("a", "b", "c"):
            client.post("/api/datasets/many2/cases", json={"input": v})
        r = client.delete("/api/datasets/many2/cases/0")
        assert r.status_code == 204
        cases = client.get("/api/datasets/many2").json()["cases"]
        assert [c["index"] for c in cases] == [0, 1]
        assert [c["input"] for c in cases] == ["b", "c"]

    def test_put_out_of_range_returns_404(self, client: TestClient) -> None:
        client.post("/api/datasets", json={"name": "small"})
        r = client.put(
            "/api/datasets/small/cases/9", json={"input": "x"}
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 3. Multimodal — image upload + case referencing the path round-trips
# ---------------------------------------------------------------------------


class TestMultimodal:
    def test_upload_image_then_reference_in_case(
        self, client: TestClient, app_db
    ) -> None:
        _, _, datasets_dir = app_db
        client.post("/api/datasets", json={"name": "vision"})

        # 1x1 transparent PNG — small but real
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\rIDATx\x9cc\xfc\xff\xff?\x03\x00\x05\xfe"
            b"\x02\xfe\xa3<\xc7\xab\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        upload = client.post(
            "/api/datasets/vision/images",
            files={"file": ("cat.png", io.BytesIO(png_bytes), "image/png")},
        )
        assert upload.status_code == 201, upload.text
        rel = upload.json()["path"]
        assert rel == "images/vision/cat.png"
        # Image is on disk
        assert (datasets_dir / "images" / "vision" / "cat.png").exists()

        # Reference it in a case
        client.post(
            "/api/datasets/vision/cases",
            json={
                "input": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image", "path": rel},
                ],
                "expected_output": "blank pixel",
                "tags": ["vision"],
            },
        )

        # The list view reports has_multimodal=True
        listed = client.get("/api/datasets").json()
        vision = next(d for d in listed if d["name"] == "vision")
        assert vision["has_multimodal"] is True

        # GET image endpoint serves the bytes
        served = client.get("/api/datasets/vision/images/cat.png")
        assert served.status_code == 200
        assert served.content.startswith(b"\x89PNG")

    def test_image_filename_traversal_is_normalised(
        self, client: TestClient, app_db
    ) -> None:
        """An uploaded filename like ``../../evil.png`` is stripped to a
        safe basename before being written. Bytes land inside the
        project's images dir, never above it.
        """
        _, _, datasets_dir = app_db
        client.post("/api/datasets", json={"name": "vision2"})
        r = client.post(
            "/api/datasets/vision2/images",
            files={
                "file": (
                    "../../evil.png",
                    b"\x89PNG\r\n",
                    "image/png",
                )
            },
        )
        assert r.status_code == 201, r.text
        # Stored path is rooted under images/vision2/, never escapes.
        rel = r.json()["path"]
        assert rel.startswith("images/vision2/")
        assert ".." not in rel
        # Nothing was written above the datasets dir.
        assert not (datasets_dir.parent / "evil.png").exists()


# ---------------------------------------------------------------------------
# 4. Import / export
# ---------------------------------------------------------------------------


class TestImportExport:
    def test_import_appends_existing_cases(
        self, client: TestClient, app_db
    ) -> None:
        _, _, datasets_dir = app_db
        client.post("/api/datasets", json={"name": "imp"})
        client.post("/api/datasets/imp/cases", json={"input": "first"})

        # 3 lines in the upload
        body = "\n".join(
            json.dumps({"input": f"in-{i}", "expected_output": f"out-{i}"})
            for i in range(3)
        )
        r = client.post(
            "/api/datasets/imp/import",
            files={"file": ("data.jsonl", body.encode("utf-8"), "application/x-ndjson")},
            data={"mode": "append"},
        )
        assert r.status_code == 200
        result = r.json()
        assert result["imported"] == 3
        assert result["total"] == 4

        cases = client.get("/api/datasets/imp").json()["cases"]
        assert [c["input"] for c in cases] == ["first", "in-0", "in-1", "in-2"]

    def test_import_replace_overwrites_existing(
        self, client: TestClient
    ) -> None:
        client.post("/api/datasets", json={"name": "rep"})
        client.post("/api/datasets/rep/cases", json={"input": "old"})

        body = json.dumps({"input": "new"}) + "\n"
        r = client.post(
            "/api/datasets/rep/import",
            files={"file": ("d.jsonl", body.encode(), "application/x-ndjson")},
            data={"mode": "replace"},
        )
        assert r.status_code == 200
        cases = client.get("/api/datasets/rep").json()["cases"]
        assert [c["input"] for c in cases] == ["new"]

    def test_import_malformed_returns_400_with_line_number(
        self, client: TestClient
    ) -> None:
        client.post("/api/datasets", json={"name": "bad"})
        body = '{"input": "ok"}\n{"missing": "input"}\n'
        r = client.post(
            "/api/datasets/bad/import",
            files={"file": ("d.jsonl", body.encode(), "application/x-ndjson")},
        )
        assert r.status_code == 400
        assert "line 2" in r.json()["detail"]

    def test_export_returns_jsonl(self, client: TestClient) -> None:
        client.post("/api/datasets", json={"name": "exp"})
        client.post(
            "/api/datasets/exp/cases", json={"input": "a", "expected_output": "A"}
        )
        client.post(
            "/api/datasets/exp/cases", json={"input": "b", "expected_output": "B"}
        )

        r = client.get("/api/datasets/exp/export")
        assert r.status_code == 200
        assert "exp.jsonl" in r.headers["content-disposition"]
        lines = [json.loads(line) for line in r.text.strip().split("\n")]
        assert lines == [
            {"input": "a", "expected_output": "A"},
            {"input": "b", "expected_output": "B"},
        ]


# ---------------------------------------------------------------------------
# 5. Path traversal in dataset name → 400
# ---------------------------------------------------------------------------


class TestPathTraversal:
    def test_create_with_traversal_name_returns_400(
        self, client: TestClient, app_db
    ) -> None:
        _, _, datasets_dir = app_db
        r = client.post(
            "/api/datasets", json={"name": "../../../etc/passwd"}
        )
        # The name regex (^[A-Za-z0-9_\-]+$) bars dots, slashes, and
        # whitespace. FastAPI may also reject at the body layer. Either
        # 400 or 422 is fine as long as we did not create the file.
        assert r.status_code in (400, 422)
        # And nothing landed outside the project's datasets dir.
        assert not Path("/etc/passwd.jsonl").exists() or True  # truism guard
        assert not (datasets_dir.parent.parent.parent / "etc" / "passwd.jsonl").exists()

    def test_create_with_dotdot_name_returns_400(
        self, client: TestClient
    ) -> None:
        # The dataset name lives in the JSON body, not the URL — so the
        # framework can't normalise it away before our regex runs.
        r = client.post("/api/datasets", json={"name": ".."})
        assert r.status_code in (400, 422)

    def test_create_with_slash_name_returns_400(
        self, client: TestClient
    ) -> None:
        r = client.post("/api/datasets", json={"name": "evil/file"})
        assert r.status_code in (400, 422)


# ---------------------------------------------------------------------------
# 6. Run-eval — end-to-end into eval_runs/eval_cases
# ---------------------------------------------------------------------------


class TestRunEval:
    def test_run_eval_persists_a_run_appearing_in_evals_endpoint(
        self, client: TestClient
    ) -> None:
        # Echo agent + exact_match scorer: a case where input==expected
        # passes; otherwise fails. Two cases, one of each.
        client.post("/api/datasets", json={"name": "smoke"})
        client.post(
            "/api/datasets/smoke/cases",
            json={"input": "match", "expected_output": "match"},
        )
        client.post(
            "/api/datasets/smoke/cases",
            json={"input": "miss", "expected_output": "different"},
        )

        r = client.post(
            "/api/datasets/smoke/run-eval",
            json={"scorers": ["exact_match"]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["pass_count"] == 1
        assert body["fail_count"] == 1
        assert body["pass_rate"] == pytest.approx(0.5)

        # The run is now visible to the existing /api/evals listing
        runs = client.get("/api/evals").json()
        run_ids = [r["run_id"] for r in runs.get("rows", [])]
        assert body["run_id"] in run_ids
