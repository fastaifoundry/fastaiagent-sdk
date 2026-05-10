"""Tests for fastaiagent._internal module."""

import os
from datetime import datetime
from enum import Enum
from uuid import uuid4

import pytest
from pydantic import BaseModel

from fastaiagent._internal.config import SDKConfig, get_config, reset_config
from fastaiagent._internal.errors import (
    AgentError,
    AgentTimeoutError,
    ChainCycleError,
    ChainError,
    FastAIAgentError,
    GuardrailBlockedError,
    LLMError,
    MaxIterationsError,
    PlatformAuthError,
    PlatformError,
    ToolError,
)
from fastaiagent._internal.serialization import from_json, serialize_value, to_json
from fastaiagent._internal.storage import SQLiteHelper

# --- Error hierarchy tests ---


class TestErrors:
    def test_base_error_is_exception(self):
        assert issubclass(FastAIAgentError, Exception)

    def test_agent_errors_inherit_from_base(self):
        assert issubclass(AgentError, FastAIAgentError)
        assert issubclass(AgentTimeoutError, AgentError)
        assert issubclass(MaxIterationsError, AgentError)

    def test_chain_errors_inherit_from_base(self):
        assert issubclass(ChainError, FastAIAgentError)
        assert issubclass(ChainCycleError, ChainError)

    def test_tool_errors_inherit_from_base(self):
        assert issubclass(ToolError, FastAIAgentError)

    def test_llm_errors_inherit_from_base(self):
        assert issubclass(LLMError, FastAIAgentError)

    def test_platform_errors_inherit_from_base(self):
        assert issubclass(PlatformError, FastAIAgentError)
        assert issubclass(PlatformAuthError, PlatformError)

    def test_guardrail_blocked_error_has_name(self):
        err = GuardrailBlockedError("no_pii", "PII detected")
        assert err.guardrail_name == "no_pii"
        assert str(err) == "PII detected"
        assert err.results == []

    def test_guardrail_blocked_error_default_message(self):
        err = GuardrailBlockedError("json_valid")
        assert "json_valid" in str(err)

    def test_catch_by_parent(self):
        with pytest.raises(FastAIAgentError):
            raise AgentTimeoutError("too slow")


# --- Config tests ---


class TestConfig:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()
        for key in list(os.environ.keys()):
            if key.startswith("FASTAIAGENT_"):
                del os.environ[key]

    def test_default_config(self):
        config = SDKConfig()
        assert config.trace_enabled is True
        assert config.local_db_path == ".fastaiagent/local.db"
        assert config.trace_db_path is None
        assert config.checkpoint_db_path is None
        assert config.prompt_dir is None
        assert config.resolved_trace_db_path == ".fastaiagent/local.db"
        assert config.resolved_checkpoint_db_path == ".fastaiagent/local.db"
        assert config.ui_enabled is False
        assert config.ui_host == "127.0.0.1"
        assert config.ui_port == 7842
        assert config.log_level == "WARNING"
        assert config.default_timeout == 120

    def test_config_from_env(self):
        os.environ["FASTAIAGENT_TRACE_ENABLED"] = "false"
        os.environ["FASTAIAGENT_LOG_LEVEL"] = "DEBUG"
        os.environ["FASTAIAGENT_DEFAULT_TIMEOUT"] = "60"
        os.environ["FASTAIAGENT_LOCAL_DB"] = "/tmp/custom.db"
        os.environ["FASTAIAGENT_UI_PORT"] = "9000"
        config = SDKConfig.from_env()
        assert config.trace_enabled is False
        assert config.log_level == "DEBUG"
        assert config.default_timeout == 60
        assert config.local_db_path == "/tmp/custom.db"
        assert config.resolved_trace_db_path == "/tmp/custom.db"
        assert config.ui_port == 9000

    def test_legacy_env_var_emits_deprecation(self):
        import warnings

        os.environ["FASTAIAGENT_TRACE_DB_PATH"] = "/tmp/legacy-traces.db"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = SDKConfig.from_env()
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        # Legacy path still honored for back-compat via the resolved helper.
        assert config.resolved_trace_db_path == "/tmp/legacy-traces.db"

    def test_get_config_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_reset_config_clears_cache(self):
        c1 = get_config()
        reset_config()
        c2 = get_config()
        assert c1 is not c2


# --- Serialization tests ---


class TestSerialization:
    def test_serialize_primitives(self):
        assert serialize_value(42) == 42
        assert serialize_value("hello") == "hello"
        assert serialize_value(True) is True
        assert serialize_value(None) is None

    def test_serialize_uuid(self):
        uid = uuid4()
        assert serialize_value(uid) == str(uid)

    def test_serialize_datetime(self):
        dt = datetime(2025, 1, 15, 10, 30, 0)
        assert serialize_value(dt) == dt.isoformat()

    def test_serialize_enum(self):
        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        assert serialize_value(Color.RED) == "red"

    def test_serialize_pydantic_model(self):
        class Item(BaseModel):
            name: str
            value: int

        result = serialize_value(Item(name="test", value=42))
        assert result == {"name": "test", "value": 42}

    def test_serialize_nested_dict(self):
        uid = uuid4()
        data = {"id": uid, "items": [1, 2, 3]}
        result = serialize_value(data)
        assert result == {"id": str(uid), "items": [1, 2, 3]}

    def test_to_json_and_from_json(self):
        data = {"key": "value", "count": 42}
        json_str = to_json(data)
        parsed = from_json(json_str)
        assert parsed == data


# --- SQLite storage tests ---


class TestSQLiteHelper:
    def test_create_and_query(self, temp_dir):
        db_path = temp_dir / "test.db"
        with SQLiteHelper(db_path) as db:
            db.execute("CREATE TABLE t (id TEXT, name TEXT)")
            db.execute("INSERT INTO t VALUES (?, ?)", ("1", "Alice"))
            rows = db.fetchall("SELECT * FROM t")
            assert len(rows) == 1
            assert rows[0]["name"] == "Alice"

    def test_fetchone(self, temp_dir):
        db_path = temp_dir / "test.db"
        with SQLiteHelper(db_path) as db:
            db.execute("CREATE TABLE t (id TEXT, name TEXT)")
            db.execute("INSERT INTO t VALUES (?, ?)", ("1", "Alice"))
            row = db.fetchone("SELECT * FROM t WHERE id = ?", ("1",))
            assert row is not None
            assert row["name"] == "Alice"

    def test_fetchone_returns_none(self, temp_dir):
        db_path = temp_dir / "test.db"
        with SQLiteHelper(db_path) as db:
            db.execute("CREATE TABLE t (id TEXT)")
            row = db.fetchone("SELECT * FROM t WHERE id = ?", ("missing",))
            assert row is None

    def test_executemany(self, temp_dir):
        db_path = temp_dir / "test.db"
        with SQLiteHelper(db_path) as db:
            db.execute("CREATE TABLE t (id TEXT, name TEXT)")
            db.executemany(
                "INSERT INTO t VALUES (?, ?)",
                [("1", "Alice"), ("2", "Bob"), ("3", "Charlie")],
            )
            rows = db.fetchall("SELECT * FROM t")
            assert len(rows) == 3

    def test_creates_parent_directories(self, temp_dir):
        db_path = temp_dir / "nested" / "deep" / "test.db"
        with SQLiteHelper(db_path) as db:
            db.execute("CREATE TABLE t (id TEXT)")
            assert db_path.parent.exists()

    def test_context_manager_closes(self, temp_dir):
        db_path = temp_dir / "test.db"
        db = SQLiteHelper(db_path)
        db.execute("CREATE TABLE t (id TEXT)")
        db.close()
        assert db._conn is None

    @pytest.mark.skipif(
        os.name == "nt", reason="POSIX file modes do not apply on Windows"
    )
    def test_chmod_owner_only_on_creation(self, temp_dir):
        """Regression for security_review_1.md H8.

        New ``local.db`` files (and the parent dir, when we created it)
        must land at owner-only perms — traces and bcrypt hashes are
        otherwise world-readable on shared hosts.
        """
        # Create a fresh directory ourselves and then a brand-new db inside
        # a brand-new subdirectory so SQLiteHelper's ``parent_was_new``
        # branch fires on the *grandchild* directory.
        new_root = temp_dir / "fresh-root"
        db_path = new_root / "data" / "test.db"
        with SQLiteHelper(db_path) as db:
            db.execute("CREATE TABLE t (id TEXT)")
        assert db_path.exists()
        # File: owner-only (0o600).
        assert (db_path.stat().st_mode & 0o777) == 0o600
        # Parent dir: owner-only (0o700).
        assert (db_path.parent.stat().st_mode & 0o777) == 0o700

    @pytest.mark.skipif(
        os.name == "nt", reason="POSIX file modes do not apply on Windows"
    )
    def test_chmod_does_not_downgrade_existing_files(self, temp_dir):
        """If a user pre-created the parent dir with their own perms, we
        must not silently rewrite them.
        """
        existing_parent = temp_dir / "user-managed"
        existing_parent.mkdir(mode=0o755)
        db_path = existing_parent / "test.db"
        # Open helper but only the file is new; the parent already existed.
        with SQLiteHelper(db_path) as db:
            db.execute("CREATE TABLE t (id TEXT)")
        # New file gets tightened, existing parent stays as the user wrote it.
        assert (db_path.stat().st_mode & 0o777) == 0o600
        assert (existing_parent.stat().st_mode & 0o777) == 0o755
