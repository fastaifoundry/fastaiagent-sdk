"""The Local UI prompt editor keeps fragments, metadata and authorship (P2, P3, 1.89.0).

P2. ``GET /api/prompts/{slug}`` returned the *resolved* template, the editor
seeded its textarea from it, and "Save as new version" PUT that text back. So
``{{@tone}} Hello {{name}}`` became a frozen copy of the fragment's text, and
editing the fragment no longer changed the prompt.

P3. The editor sent ``metadata: {}`` on every save and the route wrote it
verbatim, so each new version lost its metadata. Versions saved from the UI
were also labelled ``created_by: "code"``.

Real app (``build_app``), real SQLite, real registry. The editor gate requires
the DB to live under the cwd, so each test ``chdir``s into its project.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

RAW = "{{@tone}} Hello {{name}}"


@pytest.fixture(params=["unscoped", "project_scoped"])
def editor(request: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.prompt.registry import PromptRegistry
    from fastaiagent.ui.server import build_app

    monkeypatch.chdir(tmp_path)
    db = tmp_path / ".fastaiagent" / "local.db"
    db.parent.mkdir()
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    scoped = request.param == "project_scoped"
    if scoped:
        _project.set_project_id("proj-a")
    reg = PromptRegistry(path=str(db))
    reg.register_fragment("tone", "Be warm.")
    reg.register("greet", RAW, metadata={"owner": "ops"}, project_id="proj-a" if scoped else None)
    app = build_app(db_path=str(db), no_auth=True, project_id="proj-a" if scoped else None)
    try:
        yield TestClient(app), reg
    finally:
        _project.reset_for_testing()
        reset_config()


def test_get_returns_the_raw_template_beside_the_resolved_one(editor: Any) -> None:
    client, _ = editor
    body = client.get("/api/prompts/greet").json()
    assert body["template_raw"] == RAW
    assert body["template"] == "Be warm. Hello {{name}}"
    assert body["fragments"] == ["tone"]
    assert body["variables"] == ["name"]

    v1 = client.get("/api/prompts/greet/versions/1").json()
    assert v1["template_raw"] == RAW and v1["fragments"] == ["tone"]


def test_saving_keeps_the_fragment_reference_live(editor: Any) -> None:
    client, reg = editor
    raw = client.get("/api/prompts/greet").json()["template_raw"]

    r = client.put("/api/prompts/greet", json={"template": raw + "!", "base_version": 1})
    assert r.status_code == 200, r.text
    assert r.json()["template_raw"] == RAW + "!"

    reg.register_fragment("tone", "Be formal.")
    assert reg.load("greet").template == "Be formal. Hello {{name}}!", (
        "the saved version froze the fragment's text instead of referencing it"
    )


def test_an_edit_inherits_the_base_versions_metadata(editor: Any) -> None:
    client, _ = editor
    r = client.put("/api/prompts/greet", json={"template": RAW + "!", "base_version": 1})
    assert r.status_code == 200, r.text
    assert client.get("/api/prompts/greet/versions/2").json()["metadata"] == {"owner": "ops"}

    # Omitting base_version inherits from the latest.
    client.put("/api/prompts/greet", json={"template": RAW + "!!"})
    assert client.get("/api/prompts/greet/versions/3").json()["metadata"] == {"owner": "ops"}


def test_editing_an_older_version_inherits_that_versions_metadata(editor: Any) -> None:
    client, _ = editor
    client.put("/api/prompts/greet", json={"template": "v2", "metadata": {"owner": "growth"}})
    r = client.put("/api/prompts/greet", json={"template": "v3 from v1", "base_version": 1})
    assert r.status_code == 200, r.text
    assert client.get("/api/prompts/greet/versions/3").json()["metadata"] == {"owner": "ops"}


def test_an_explicit_empty_metadata_clears_it(editor: Any) -> None:
    client, _ = editor
    client.put("/api/prompts/greet", json={"template": "x", "metadata": {}})
    assert client.get("/api/prompts/greet/versions/2").json()["metadata"] == {}


def test_a_missing_base_version_is_404(editor: Any) -> None:
    client, _ = editor
    r = client.put("/api/prompts/greet", json={"template": "x", "base_version": 9})
    assert r.status_code == 404


def test_ui_versions_are_labelled_ui_and_listed_in_one_shape(editor: Any) -> None:
    client, _ = editor
    client.put("/api/prompts/greet", json={"template": RAW + "!", "base_version": 1})

    versions = client.get("/api/prompts/greet/versions").json()["versions"]
    assert [v["created_by"] for v in versions] == ["code", "ui"]
    for v in versions:
        assert isinstance(v["variables"], list), "variables came back as a JSON string"
        assert v["variables"] == ["name"]
        assert v["template"].startswith("Be warm. Hello")
        assert v["template_raw"].startswith("{{@tone}}")
        assert v["fragments"] == ["tone"]
