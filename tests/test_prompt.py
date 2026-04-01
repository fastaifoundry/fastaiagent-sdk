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
