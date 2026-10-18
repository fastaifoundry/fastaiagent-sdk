"""Gemini 3.5+ is priced and is the default; a Bedrock client keeps its region (1.91.0).

* Gemini 3.x / 4 had no rates, so every run on a current Gemini model reported
  ``cost_known=False`` — and ``cost_limit`` (fail-closed on unknown spend) blocked
  it. Rates are Google's published list prices (ai.google.dev/gemini-api/docs/pricing,
  read 2026-10-18), including the increases Google has already announced: a rate
  can carry the date it takes effect, so a cost computed after that date uses it
  without an SDK release.
* The Gemini preset's default model moves from ``gemini-2.5-flash`` to
  ``gemini-3.8-flash``.
* **X9** — ``LLMClient.to_dict()`` dropped a Bedrock client's ``region``, so a
  replayed or plane-dispatched Bedrock agent silently ran in us-east-1. Only that
  one allowlisted key is serialized; other constructor extras stay local.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from fastaiagent._internal.pricing import compute_cost_usd, usage_cost
from fastaiagent.llm.client import LLMClient

M = 1_000_000


@pytest.mark.parametrize(
    ("model", "on", "inp", "out"),
    [
        ("gemini-4-argon", date(2026, 10, 18), 2.00, 10.00),
        ("gemini-4-argon", date(2026, 11, 30), 2.00, 10.00),
        ("gemini-4-argon", date(2026, 12, 1), 4.00, 20.00),
        ("gemini-3.8-flash", date(2026, 12, 31), 0.75, 3.75),
        ("gemini-3.8-flash", date(2027, 1, 1), 1.50, 7.50),
        ("gemini-3.6-flash", date(2026, 10, 18), 0.75, 3.75),
        ("gemini-3.6-flash", date(2027, 2, 1), 1.50, 7.50),
        ("gemini-3.5-flash-lite", date(2026, 10, 18), 0.30, 2.50),
        ("gemini-3.1-pro-preview", date(2026, 10, 18), 2.00, 12.00),
        ("gemini-3.1-flash-lite", date(2026, 10, 18), 0.25, 1.50),
        ("gemini-3-flash-preview", date(2026, 10, 18), 0.50, 3.00),
    ],
)
def test_gemini_rates_follow_the_published_schedule(model, on, inp, out) -> None:
    assert compute_cost_usd(model, M, 0, on=on) == pytest.approx(inp)
    assert compute_cost_usd(model, 0, M, on=on) == pytest.approx(out)


def test_a_dated_rate_applies_to_versioned_ids_of_the_model() -> None:
    assert compute_cost_usd("gemini-4-argon-001", M, 0, on=date(2026, 12, 2)) == pytest.approx(4.0)


def test_unpublished_gemini_models_stay_unpriced() -> None:
    """Callable via the API but absent from Google's pricing page: unknown is
    honest, a guessed rate is not."""
    for model in ("gemini-3.5-flash", "gemini-3.7-flash", "gemini-flash-latest"):
        assert usage_cost(model, {"prompt_tokens": 10, "completion_tokens": 5}) == (0.0, False)


def test_undated_rates_are_unchanged() -> None:
    assert compute_cost_usd("gemini-2.5-flash", M, M) == pytest.approx(0.30 + 2.50)
    assert compute_cost_usd("gpt-5-mini", M, M, on=date(2030, 1, 1)) == pytest.approx(0.25 + 2.00)


def test_today_is_the_default_date() -> None:
    today = datetime.now(timezone.utc).date()
    assert compute_cost_usd("gemini-4-argon", M, 0) == compute_cost_usd(
        "gemini-4-argon", M, 0, on=today
    )


def test_gemini_preset_defaults_to_3_8_flash(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert LLMClient(provider="gemini").model == "gemini-3.8-flash"


# ── X9 ─────────────────────────────────────────────────────────────────────


def test_bedrock_region_survives_a_round_trip() -> None:
    client = LLMClient(provider="bedrock", model="anthropic.claude-3-haiku", region="eu-west-1")
    data = client.to_dict()
    assert data["region"] == "eu-west-1"
    assert LLMClient.from_dict(data)._extra.get("region") == "eu-west-1"


def test_a_client_without_a_region_serializes_exactly_as_before() -> None:
    assert "region" not in LLMClient(provider="bedrock", model="m").to_dict()
    assert "region" not in LLMClient(provider="openai", model="gpt-5-mini", api_key="k").to_dict()


def test_other_constructor_extras_are_not_serialized() -> None:
    """Allowlist, not denylist: an arbitrary extra may be a credential."""
    data = LLMClient(
        provider="bedrock", model="m", region="eu-west-1", aws_secret_access_key="SECRET"
    ).to_dict()
    assert "SECRET" not in repr(data)
    assert set(data) == {"provider", "model", "region"}
