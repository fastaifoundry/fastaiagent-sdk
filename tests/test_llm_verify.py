"""Tests for ``LLMClient`` TLS verification config (``verify`` / env fallback).

No network, no mocks — these exercise the real httpx + ssl stack to assert that
the ``verify`` setting actually reaches the transport's SSL context. This is the
knob that lets the SDK talk to a provider behind a corporate gateway (e.g. Azure
OpenAI on Azure ML) the same way ``openai``'s ``http_client=httpx.Client(...)``
does.
"""

from __future__ import annotations

import ssl
import warnings

import certifi
import pytest

from fastaiagent.llm.client import LLMClient


def _ssl_context(client: LLMClient):
    """Pull the ssl.SSLContext httpx built for this client's transport."""
    http = client._new_async_client()
    return http._transport._pool._ssl_context


def test_default_verify_true_uses_public_roots() -> None:
    ctx = _ssl_context(LLMClient(provider="azure", model="m", api_key="k"))
    assert ctx is not None
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    # certifi ships > 100 public roots; the default trust store must be populated.
    assert len(ctx.get_ca_certs()) > 50


def test_verify_false_disables_verification_and_warns() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client = LLMClient(provider="azure", model="m", api_key="k", verify=False)
    assert any("verification is disabled" in str(w.message) for w in caught)

    ctx = _ssl_context(client)
    assert ctx.verify_mode == ssl.CERT_NONE
    assert ctx.check_hostname is False


def test_verify_path_loads_that_bundle(tmp_path) -> None:
    # A one-cert bundle: prove the SSL context trusts exactly what we point at.
    one = tmp_path / "one.pem"
    data = certifi.contents()
    first = data.split("-----END CERTIFICATE-----")[0] + "-----END CERTIFICATE-----\n"
    one.write_text(first)

    ctx = _ssl_context(
        LLMClient(provider="azure", model="m", api_key="k", verify=str(one))
    )
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert len(ctx.get_ca_certs()) == 1


def test_verify_accepts_custom_ssl_context() -> None:
    custom = ssl.create_default_context()
    client = LLMClient(provider="azure", model="m", api_key="k", verify=custom)
    # The exact object is reused, not rebuilt.
    assert _ssl_context(client) is custom


def test_env_fallback_disables_when_param_default(monkeypatch) -> None:
    monkeypatch.setenv("FASTAIAGENT_LLM_VERIFY", "false")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = LLMClient(provider="azure", model="m", api_key="k")
    assert _ssl_context(client).verify_mode == ssl.CERT_NONE


def test_env_fallback_accepts_bundle_path(monkeypatch, tmp_path) -> None:
    one = tmp_path / "one.pem"
    data = certifi.contents()
    one.write_text(
        data.split("-----END CERTIFICATE-----")[0] + "-----END CERTIFICATE-----\n"
    )
    monkeypatch.setenv("FASTAIAGENT_LLM_VERIFY", str(one))
    ctx = _ssl_context(LLMClient(provider="azure", model="m", api_key="k"))
    assert len(ctx.get_ca_certs()) == 1


@pytest.mark.parametrize("truthy", ["true", "1", "yes"])
def test_env_truthy_keeps_verification(monkeypatch, truthy) -> None:
    monkeypatch.setenv("FASTAIAGENT_LLM_VERIFY", truthy)
    ctx = _ssl_context(LLMClient(provider="azure", model="m", api_key="k"))
    assert ctx.verify_mode == ssl.CERT_REQUIRED


def test_explicit_param_overrides_with_false(monkeypatch) -> None:
    # An explicit verify=False must disable even if env says otherwise is moot;
    # the point: passing verify=False always yields CERT_NONE.
    monkeypatch.delenv("FASTAIAGENT_LLM_VERIFY", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = LLMClient(provider="openai", model="gpt-4o", api_key="k", verify=False)
    assert _ssl_context(client).verify_mode == ssl.CERT_NONE
