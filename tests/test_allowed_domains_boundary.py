"""``allowed_domains`` matches the URL's host, not a suffix of it (1.90.0, F1).

Until 1.90.0 the check was ``url_domain.endswith(d)`` on the text between ``://``
and the next ``/``. That let a look-alike domain through (``evilexample.com``
ends with ``example.com``), blocked the allowed host itself when the URL named a
port (``example.com:443``), and an empty entry allowed every URL, because every
string ends with ``""``.
"""

from __future__ import annotations

import pytest

from fastaiagent.guardrail import allowed_domains


def _passes(domains: list[str], text: str) -> bool:
    return allowed_domains(domains).fn(text).passed  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "url",
    [
        "https://evilexample.com/x",
        "https://example.com.evil.net/x",
        "http://notexample.com",
        # userinfo: the host is evil.com, whatever comes before the @
        "https://example.com@evil.com/x",
    ],
)
def test_look_alike_hosts_are_blocked(url: str) -> None:
    assert not _passes(["example.com"], f"fetch {url} now")


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/x",
        "https://api.example.com/v1",
        "https://example.com:443/x",
        "https://EXAMPLE.com/x",
        "https://example.com./x",
        "https://user@example.com/x",
    ],
)
def test_the_allowed_host_and_its_subdomains_pass(url: str) -> None:
    assert _passes(["example.com"], f"fetch {url} now")


def test_entries_are_case_insensitive_and_accept_a_wildcard_prefix() -> None:
    assert _passes(["Example.COM"], "https://api.example.com/x")
    assert _passes(["*.example.com"], "https://api.example.com/x")
    assert _passes([".example.com"], "https://example.com/x")
    assert not _passes(["*.example.com"], "https://evilexample.com/x")


def test_every_url_in_the_text_is_judged() -> None:
    result = allowed_domains(["example.com"]).fn(
        "see https://example.com/a and https://evilexample.com/b"
    )
    assert result.passed is False
    assert result.metadata["blocked_domains"] == ["evilexample.com"]


def test_a_bare_domain_inside_a_json_tool_call_is_judged_by_its_host() -> None:
    # The guardrail's default position is tool_call, which judges a JSON envelope.
    # 1.89.0 captured ``example.com"}}`` here and blocked the allowed host.
    import json

    envelope = json.dumps({"tool": "fetch", "arguments": {"url": "https://example.com"}})
    assert _passes(["example.com"], envelope)
    evil = json.dumps({"tool": "fetch", "arguments": {"url": "https://evilexample.com"}})
    assert not _passes(["example.com"], evil)
    assert _passes(["example.com"], "Read it at https://example.com, then reply.")


def test_an_entry_with_a_port_allows_that_port_only() -> None:
    # 1.89.0 behaviour kept: a port in the entry matched the same port.
    assert _passes(["example.com:8080"], "https://example.com:8080/x")
    assert not _passes(["example.com:8080"], "https://example.com:9090/x")
    assert not _passes(["example.com:8080"], "https://evilexample.com:8080/x")


@pytest.mark.parametrize("entry", ["", "  ", "*.", "."])
def test_a_blank_entry_is_refused_at_construction(entry: str) -> None:
    # A blank entry used to allow every URL.
    with pytest.raises(ValueError, match="allowed_domains"):
        allowed_domains(["example.com", entry])
