"""Tests for fastaiagent.prompt module."""

from __future__ import annotations

import pytest

from fastaiagent._internal.errors import PromptNotFoundError
from fastaiagent.prompt import Fragment, Prompt, PromptRegistry


class TestPrompt:
    def test_creation(self):
        p = Prompt(name="greet", template="Hello {{name}}!")
        assert p.name == "greet"
        assert "name" in p.variables

    def test_format(self):
        p = Prompt(name="greet", template="Hello {{name}}, welcome to {{place}}!")
        result = p.format(name="Alice", place="Wonderland")
        assert result == "Hello Alice, welcome to Wonderland!"

    def test_variable_extraction(self):
        p = Prompt(name="test", template="{{a}} and {{b}} and {{a}}")
        assert sorted(p.variables) == ["a", "b"]

    def test_to_dict_roundtrip(self):
        original = Prompt(name="test", template="Hello {{name}}", version=2)
        d = original.to_dict()
        restored = Prompt.from_dict(d)
        assert restored.name == original.name
        assert restored.template == original.template
        assert restored.version == original.version


class TestFragment:
    def test_creation(self):
        f = Fragment(name="tone", content="Be professional and concise.")
        assert f.name == "tone"

    def test_to_dict_roundtrip(self):
        original = Fragment(name="tone", content="Be helpful.", version=2)
        d = original.to_dict()
        restored = Fragment.from_dict(d)
        assert restored.name == original.name
        assert restored.content == original.content


class TestPromptRegistry:
    def test_register_and_load(self, temp_dir):
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register("greet", "Hello {{name}}!")
        prompt = reg.load("greet")
        assert prompt.name == "greet"
        assert prompt.format(name="World") == "Hello World!"

    def test_versioning(self, temp_dir):
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register("greet", "Hello {{name}}!", version=1)
        reg.register("greet", "Hi {{name}}!", version=2)

        v1 = reg.load("greet", version=1)
        v2 = reg.load("greet", version=2)
        assert v1.template == "Hello {{name}}!"
        assert v2.template == "Hi {{name}}!"

    def test_alias(self, temp_dir):
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register("greet", "Hello {{name}}!", version=1)
        reg.register("greet", "Hi {{name}}!", version=2)
        reg.set_alias("greet", 1, "production")

        prod = reg.load("greet", alias="production")
        assert prod.version == 1

    def test_fragment_composition(self, temp_dir):
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register_fragment("tone", "Be professional and concise.")
        reg.register("support", "Help the user. {{@tone}}")

        prompt = reg.load("support")
        assert "professional" in prompt.template

    def test_list(self, temp_dir):
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register("a", "template a")
        reg.register("b", "template b")
        prompts = reg.list()
        names = [p["name"] for p in prompts]
        assert "a" in names
        assert "b" in names

    def test_diff(self, temp_dir):
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register("test", "Version one", version=1)
        reg.register("test", "Version two", version=2)
        diff = reg.diff("test", 1, 2)
        assert "Version one" in diff
        assert "Version two" in diff

    def test_not_found(self, temp_dir):
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        with pytest.raises(PromptNotFoundError):
            reg.load("nonexistent")

    # --- Versioning edge cases ---

    def test_auto_increment_versioning(self, temp_dir):
        """register() without explicit version should auto-increment."""
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        p1 = reg.register("greet", "Hello {{name}}!")
        p2 = reg.register("greet", "Hi {{name}}!")
        p3 = reg.register("greet", "Hey {{name}}!")
        assert p1.version == 1
        assert p2.version == 2
        assert p3.version == 3

    def test_load_latest_version(self, temp_dir):
        """load() with no version returns the most recent."""
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register("greet", "v1 {{name}}")
        reg.register("greet", "v2 {{name}}")
        reg.register("greet", "v3 {{name}}")
        latest = reg.load("greet")
        assert latest.version == 3
        assert latest.template == "v3 {{name}}"

    def test_forced_version_gap(self, temp_dir):
        """Forcing version=5 after v1 should work; load() returns v5 as latest."""
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register("greet", "Hello {{name}}!")
        reg.register("greet", "Yo {{name}}!", version=5)
        latest = reg.load("greet")
        assert latest.version == 5
        v1 = reg.load("greet", version=1)
        assert v1.template == "Hello {{name}}!"

    def test_load_nonexistent_version(self, temp_dir):
        """Loading a version that doesn't exist raises PromptNotFoundError."""
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register("greet", "Hello!")
        with pytest.raises(PromptNotFoundError):
            reg.load("greet", version=99)

    # --- Alias edge cases ---

    def test_set_alias_nonexistent_prompt(self, temp_dir):
        """set_alias on a prompt that doesn't exist raises PromptNotFoundError."""
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        with pytest.raises(PromptNotFoundError):
            reg.set_alias("missing", 1, "production")

    def test_load_nonexistent_alias(self, temp_dir):
        """Loading with an alias that doesn't exist raises PromptNotFoundError."""
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register("greet", "Hello!")
        with pytest.raises(PromptNotFoundError):
            reg.load("greet", alias="nonexistent")

    # --- Fragment edge cases ---

    def test_unresolved_fragment_stays(self, temp_dir):
        """{{@missing}} stays as-is when the fragment is not registered."""
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register("support", "Help the user. {{@missing}}")
        prompt = reg.load("support")
        assert "{{@missing}}" in prompt.template

    def test_multiple_fragments_in_template(self, temp_dir):
        """Multiple fragment references in one template all resolve."""
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register_fragment("tone", "Be professional.")
        reg.register_fragment("format", "Use bullet points.")
        reg.register("support", "{{@tone}} Help the user. {{@format}}")
        prompt = reg.load("support")
        assert "Be professional." in prompt.template
        assert "Use bullet points." in prompt.template
        assert "{{@tone}}" not in prompt.template
        assert "{{@format}}" not in prompt.template

    def test_fragment_overwrite(self, temp_dir):
        """Re-registering a fragment with the same name overwrites content."""
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register_fragment("tone", "Be casual.")
        reg.register_fragment("tone", "Be formal.")
        reg.register("support", "{{@tone}}")
        prompt = reg.load("support")
        assert "Be formal." in prompt.template
        assert "Be casual." not in prompt.template

    # --- Other edge cases ---

    def test_load_returns_latest_version_number(self, temp_dir):
        """load() returns a prompt whose .version reflects the latest."""
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register("greet", "v1")
        reg.register("greet", "v2")
        reg.register("greet", "v3")
        assert reg.load("greet").version == 3

    def test_list_shows_latest_version_and_count(self, temp_dir):
        """list() returns latest_version and total version count per prompt."""
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register("greet", "v1")
        reg.register("greet", "v2")
        reg.register("greet", "v3")
        reg.register("other", "only one")
        prompts = reg.list()
        by_name = {p["name"]: p for p in prompts}
        assert by_name["greet"]["latest_version"] == 3
        assert by_name["greet"]["versions"] == 3
        assert by_name["other"]["latest_version"] == 1
        assert by_name["other"]["versions"] == 1

    def test_empty_list(self, temp_dir):
        """list() on a fresh registry returns an empty list."""
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        assert reg.list() == []

    def test_diff_no_changes(self, temp_dir):
        """Diff between two versions with identical templates shows no changes."""
        reg = PromptRegistry(path=str(temp_dir / "prompts"))
        reg.register("test", "Same template", version=1)
        reg.register("test", "Same template", version=2)
        diff = reg.diff("test", 1, 2)
        assert "(no template changes)" in diff

    def test_format_missing_variable(self):
        """format() without all variables leaves the placeholder."""
        p = Prompt(name="greet", template="Hello {{name}}, welcome to {{place}}!")
        result = p.format(name="Alice")
        assert "Alice" in result
        assert "{{place}}" in result

    def test_prompt_no_variables(self):
        """Template with no {{}} has empty variables list."""
        p = Prompt(name="static", template="No variables here.")
        assert p.variables == []

    # --- SQLite-backed registry specifics (post-YAML migration) ---

    def test_registry_accepts_db_file_path(self, temp_dir):
        db = temp_dir / "local.db"
        reg = PromptRegistry(path=str(db))
        reg.register("greet", "Hello {{name}}!")
        assert db.exists()
        assert reg.load("greet").template == "Hello {{name}}!"

    def test_registry_accepts_directory_path(self, temp_dir):
        dir_path = temp_dir / "prompts"
        reg = PromptRegistry(path=str(dir_path))
        reg.register("greet", "Hello!")
        # Legacy dir form places local.db inside the directory.
        assert (dir_path / "local.db").exists()

    def test_is_local_when_file_inside_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        reg = PromptRegistry(path=str(tmp_path / "local.db"))
        assert reg.is_local() is True

    def test_is_local_false_when_file_outside_cwd(self, tmp_path, monkeypatch):
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        reg = PromptRegistry(path=str(outside / "local.db"))
        assert reg.is_local() is False

    def test_prompts_share_local_db_with_other_stores(self, temp_dir):
        """Prompts, traces, checkpoints all land in one SQLite file."""
        db = temp_dir / "local.db"
        reg = PromptRegistry(path=str(db))
        reg.register("hello", "Hello {{name}}!")

        from fastaiagent._internal.storage import SQLiteHelper

        with SQLiteHelper(db) as raw:
            tables = {
                r["name"]
                for r in raw.fetchall(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert {"prompts", "prompt_versions", "spans", "checkpoints"}.issubset(tables)
