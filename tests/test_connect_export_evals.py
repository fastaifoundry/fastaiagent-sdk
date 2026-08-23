"""``connect(export_evals=...)`` posture + enroll attestation (Part D).

Eval export is *egress*, so the local flag is final — the plane cannot override
it. What the plane gets instead is the posture, attested at enroll, so
"export disabled" stays distinct from "this team isn't running evals".
"""

from __future__ import annotations

import pytest

from fastaiagent.client import _connection
from fastaiagent.eval.platform_export import eval_export_enabled

# Port 1 is reserved: connect() takes its real unreachable-plane path.
_UNREACHABLE = "http://127.0.0.1:1"


@pytest.fixture(autouse=True)
def _reset_connection():
    saved = (
        _connection.api_key,
        _connection.target,
        _connection.project,
        _connection.project_id,
        _connection.export_evals,
        _connection._eval_processor,
    )
    yield
    (
        _connection.api_key,
        _connection.target,
        _connection.project,
        _connection.project_id,
        _connection.export_evals,
        _connection._eval_processor,
    ) = saved


class TestPosture:
    def test_disconnected_is_always_off(self):
        _connection.api_key = None
        assert eval_export_enabled() is False

    def test_connected_unset_defaults_on(self):
        _connection.api_key = "k"
        _connection.export_evals = None
        assert eval_export_enabled() is True

    def test_explicit_false_is_honored(self):
        _connection.api_key = "k"
        _connection.export_evals = False
        assert eval_export_enabled() is False

    def test_kwarg_beats_env(self, monkeypatch):
        _connection.api_key = "k"
        monkeypatch.setenv("FASTAIAGENT_EXPORT_EVALS", "1")
        _connection.export_evals = False
        assert eval_export_enabled() is False

    def test_env_used_when_unset(self, monkeypatch):
        _connection.api_key = "k"
        _connection.export_evals = None
        monkeypatch.setenv("FASTAIAGENT_EXPORT_EVALS", "0")
        assert eval_export_enabled() is False


class TestConnectWiring:
    def test_connect_records_posture_and_disconnect_resets(self):
        import fastaiagent as fa

        try:
            fa.connect(api_key="k", target=_UNREACHABLE, export_evals=False)
        except Exception:
            pass  # auth check can't reach the plane; posture is set before that
        assert _connection.export_evals is False
        fa.disconnect()
        assert _connection.export_evals is None
        assert _connection._eval_processor is None


class TestEnrollAttestation:
    """Real localhost server, real httpx — no mocking (repo rule)."""

    @staticmethod
    def _enroll_bodies(server) -> list:
        return [r["body"] for r in server.requests if r["path"].endswith("/governance/enroll")]

    def test_enroll_body_carries_export_evals(self, capture_server, isolated_local_db):
        """The marking mechanism: the plane learns the posture at enroll."""
        from fastaiagent import governance

        _connection.api_key = "k"
        _connection.target = capture_server.url

        _connection.export_evals = False
        governance.enroll()
        assert self._enroll_bodies(capture_server)[-1]["export_evals"] is False

        _connection.export_evals = True
        governance.enroll()
        assert self._enroll_bodies(capture_server)[-1]["export_evals"] is True

    def test_enroll_still_carries_existing_keys(self, capture_server, isolated_local_db):
        """The new key is additive — it must not displace the WS4 payload."""
        from fastaiagent import governance

        _connection.api_key = "k"
        _connection.target = capture_server.url
        _connection.export_evals = True
        governance.enroll()

        body = self._enroll_bodies(capture_server)[-1]
        assert {"instance_id", "sdk_version", "fail_mode", "protocol_version"} <= set(body)
