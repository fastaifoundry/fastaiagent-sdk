"""Every path-shaped environment variable expands ``~`` and ``$VARS``.

The symptom this closes is not cosmetic. ``FASTAIAGENT_LOCAL_DB=~/x/local.db``
used to be passed through a bare ``str``, so SQLite created a **literal** ``./~/``
directory relative to whatever the process's current working directory happened
to be. Two processes started from two directories therefore got two different
stores — and a ``resume`` from the second found nothing, with no error to explain
it. ``test_a_tilde_path_resolves_to_the_same_store_from_two_directories`` is that
symptom, written down.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fastaiagent._internal.config import SDKConfig, reset_config
from fastaiagent._internal.storage import SQLiteHelper

#: field on ``SDKConfig`` -> the environment variable that sets it.
PATH_VARS: dict[str, str] = {
    "local_db_path": "FASTAIAGENT_LOCAL_DB",
    "trace_db_path": "FASTAIAGENT_TRACE_DB_PATH",
    "checkpoint_db_path": "FASTAIAGENT_CHECKPOINT_DB_PATH",
    "prompt_dir": "FASTAIAGENT_PROMPT_DIR",
    "cache_dir": "FASTAIAGENT_CACHE_DIR",
    "kb_dir": "FASTAIAGENT_KB_DIR",
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway ``$HOME`` so a ``~`` path cannot touch the real one."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))  # Windows
    reset_config()
    yield fake_home
    reset_config()


@pytest.fixture
def in_tmp_cwd(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


@pytest.mark.parametrize("field,var", sorted(PATH_VARS.items()))
def test_tilde_expands_and_no_literal_tilde_directory_appears(
    field, var, home, in_tmp_cwd, monkeypatch, recwarn
):
    monkeypatch.setenv(var, "~/store/thing.db")
    cfg = SDKConfig.from_env()
    value = getattr(cfg, field)
    assert value is not None
    assert Path(value).is_absolute(), f"{var} did not expand to an absolute path: {value!r}"
    assert str(value).startswith(str(home)), f"{var} resolved outside $HOME: {value!r}"
    assert not (in_tmp_cwd / "~").exists()


@pytest.mark.parametrize("field,var", sorted(PATH_VARS.items()))
def test_dollar_home_expands(field, var, home, in_tmp_cwd, monkeypatch, recwarn):
    monkeypatch.setenv(var, "$HOME/store/thing.db")
    cfg = SDKConfig.from_env()
    value = getattr(cfg, field)
    assert str(value).startswith(str(home)), f"{var} did not expand $HOME: {value!r}"
    assert "$HOME" not in str(value)


@pytest.mark.parametrize("field,var", sorted(PATH_VARS.items()))
def test_empty_value_falls_back_to_the_default(field, var, home, monkeypatch, recwarn):
    monkeypatch.setenv(var, "   ")
    assert getattr(SDKConfig.from_env(), field) == getattr(SDKConfig(), field)


def test_sqlite_helper_expands_directly_supplied_tilde_paths(home, in_tmp_cwd):
    """``fastaiagent ui --db ~/x.db`` and friends bypass ``SDKConfig`` entirely."""
    db = SQLiteHelper("~/direct/local.db")
    try:
        db.execute("CREATE TABLE IF NOT EXISTS t (id TEXT)")
        db.execute("INSERT INTO t VALUES (?)", ("row",))
        assert db.db_path.is_absolute()
        assert str(db.db_path).startswith(str(home))
    finally:
        db.close()
    assert (home / "direct" / "local.db").exists()
    assert not (in_tmp_cwd / "~").exists()


def test_model_catalog_path_expands(home, in_tmp_cwd, monkeypatch):
    from fastaiagent.ui.model_catalog import catalog_path

    monkeypatch.setenv("FASTAIAGENT_MODEL_CATALOG", "~/catalog/models.json")
    path = catalog_path(None)
    assert path.is_absolute() and str(path).startswith(str(home))


def test_kb_root_expands(home, in_tmp_cwd, monkeypatch):
    from fastaiagent.ui.routes.kb import kb_root

    monkeypatch.setenv("FASTAIAGENT_KB_DIR", "~/kbs")
    reset_config()
    assert str(kb_root()) == str(home / "kbs")


def test_llm_verify_ca_bundle_path_expands(home, monkeypatch):
    """A CA bundle is a path too — ``~/certs/corp.pem`` must resolve."""
    import ssl

    # A real CA file, so ``ssl.create_default_context(cafile=...)`` succeeds and
    # the only thing under test is whether the ``~`` resolved.
    real = Path(ssl.get_default_verify_paths().openssl_cafile or "")
    if not real.exists():  # pragma: no cover — environment-dependent
        pytest.skip("no system CA bundle to copy")
    bundle = home / "certs" / "corp.pem"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(real.read_bytes())

    from fastaiagent.llm.client import LLMClient

    monkeypatch.setenv("FASTAIAGENT_LLM_VERIFY", "~/certs/corp.pem")
    client = LLMClient(provider="openai", model="gpt-4o", api_key="k")
    assert isinstance(client._verify, ssl.SSLContext)


def test_a_tilde_path_resolves_to_the_same_store_from_two_directories(home, tmp_path, monkeypatch):
    """The symptom, named: write a checkpoint from cwd A, resume from cwd B.

    Both processes are configured identically (``FASTAIAGENT_LOCAL_DB=~/…``).
    Before the fix each one created its own ``./~/`` tree, so the second found
    no execution at all.
    """
    from fastaiagent.chain.checkpoint import Checkpoint
    from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", "~/shared/local.db")
    reset_config()

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    from fastaiagent._internal.config import get_config

    monkeypatch.chdir(dir_a)
    store_a = SQLiteCheckpointer(db_path=get_config().resolved_checkpoint_db_path)
    store_a.setup()
    store_a.put(
        Checkpoint(
            execution_id="ex-cross-cwd",
            chain_name="cross-cwd",
            node_id="turn:0",
            status="interrupted",
            state_snapshot={"canary": "from-dir-a"},
        )
    )
    store_a.close()
    assert not (dir_a / "~").exists(), "cwd A still created a literal ~ directory"

    monkeypatch.chdir(dir_b)
    reset_config()
    store_b = SQLiteCheckpointer(db_path=get_config().resolved_checkpoint_db_path)
    store_b.setup()
    found = store_b.get_last("ex-cross-cwd")
    store_b.close()
    assert not (dir_b / "~").exists(), "cwd B still created a literal ~ directory"

    assert found is not None, "the checkpoint written from cwd A was invisible from cwd B"
    assert found.state_snapshot["canary"] == "from-dir-a"
    assert (home / "shared" / "local.db").exists()
    assert os.path.isabs(get_config().resolved_checkpoint_db_path)
