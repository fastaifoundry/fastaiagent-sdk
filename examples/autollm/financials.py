"""AutoLLM on a real extraction task — pulling values from financial statements.

Classification demos (see ``agent.py``) undersell AutoLLM: the win is a format
fix and the prompt barely changes. This example is the task people actually have —
extract a critical value from a financial statement table — and it shows AutoLLM
recovering a genuine *output convention* a strong model can't guess:

    - Apply the scale in the header: "(in thousands)" -> x1,000, "(in millions)" -> x1,000,000
    - Accounting negatives: a value in parentheses "(1,120)" is NEGATIVE
    - Answer with the number itself, not a sentence

On **gpt-4o** — a strong model — the baseline prompt scores **0%**: it reads the
tables perfectly but answers with the number *as printed* ("(1,120) thousand"),
never applying the scale, so it's the wrong magnitude. AutoLLM reads the failures
(now including each case's *expected* value, as of 1.38.0), recovers the
convention, and the same model jumps to a holdout-confirmed win.

Grading uses a small ``NumericMatch`` scorer that compares the *value* (magnitude +
sign), tolerant of commas / ``$`` but not of an un-applied scale — the substance an
analyst cares about, not punctuation.

**Requires FastAIAgent >= 1.38.0** (the proposer must see expected values to learn
an extraction convention). Real OpenAI calls, no mocks.

Run
---
    export OPENAI_API_KEY=sk-...
    # from the repo root, so the run lands in the local.db `fastaiagent ui` reads
    python examples/autollm/financials.py
    fastaiagent ui            # open -> AutoLLM -> the "autollm financials demo" run
"""

from __future__ import annotations

import re
from typing import Any

import fastaiagent as fa
from fastaiagent.eval.scorer import Scorer, ScorerResult


class NumericMatch(Scorer):
    """Grade the extracted *number* (magnitude + sign), not its punctuation.

    ``-155,000,000`` and ``-155000000`` both pass; ``155`` for an expected
    ``-155000000`` is the wrong magnitude and fails — so the model must actually
    apply the thousands/millions scale.
    """

    name = "numeric_match"

    @staticmethod
    def _parse(text: str | None) -> float | None:
        if text is None:
            return None
        s = str(text).strip()
        m = re.search(r"\(?\s*-?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*\)?", s)
        if not m:
            return None
        try:
            value = float(m.group(1).replace(",", ""))
        except ValueError:
            return None
        span = s[m.start():m.end()]
        if "(" in span or "-" in span:
            value = -value
        return value

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        if expected is None:
            return ScorerResult(score=0.0, passed=False, reason="No expected value")
        exp, got = self._parse(expected), self._parse(output)
        if exp is None or got is None:
            return ScorerResult(score=0.0, passed=False, reason="Unparseable number")
        passed = abs(got - exp) <= max(1.0, abs(exp) * 1e-6)
        return ScorerResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=None if passed else f"got {got:.0f}, expected {exp:.0f}",
        )


_FACTOR = {"thousands": 1000, "millions": 1_000_000}


def _disp(v: int) -> str:
    return f"({abs(v):,})" if v < 0 else f"{v:,}"


def mk(company: str, scale: str, row: str, year: int, v2023: int, v2022: int) -> dict[str, str]:
    """One compact, question-first extraction case (kept short so the proposer sees
    the whole thing). Gold applies the scale to the asked year's printed figure."""
    val = v2023 if year == 2023 else v2022
    table = (
        f"Q: {row} for {year}?\n"
        f"{company} (in {scale})       2023        2022\n"
        f"{row}   {_disp(v2023)}   {_disp(v2022)}"
    )
    return {"input": table, "expected_output": str(val * _FACTOR[scale])}


DATASET: list[dict[str, str]] = [
    mk("NOVA INDUSTRIES", "thousands", "Net income (loss)", 2022, 3210, -1120),
    mk("NOVA INDUSTRIES", "thousands", "Revenue", 2023, 45231, 39880),
    mk("BEACON SOFTWARE", "thousands", "Operating income", 2023, 8940, 4120),
    mk("CIRRUS MATERIALS", "thousands", "Gross profit", 2023, 22455, 19200),
    mk("DELTA FOODS", "thousands", "Net income", 2023, 5013, 4280),
    mk("EVERGREEN ENERGY", "thousands", "Total assets", 2023, 128900, 119400),
    mk("EVERGREEN ENERGY", "thousands", "Total liabilities", 2023, 76300, 71900),
    mk("FOXTROT RETAIL", "thousands", "Cash and equivalents", 2023, 14205, 11880),
    mk("FOXTROT RETAIL", "thousands", "Total equity (deficit)", 2022, 9100, -3400),
    mk("GLOBAL TELECOM", "millions", "Total revenue", 2023, 12450, 11900),
    mk("GLOBAL TELECOM", "millions", "Net income", 2023, 1830, 1540),
    mk("HELIO PHARMA", "millions", "Total assets", 2023, 45900, 42100),
    mk("HELIO PHARMA", "millions", "Long-term debt", 2023, 8250, 9000),
    mk("IONIX SEMI", "millions", "Operating income (loss)", 2022, 410, -640),
    mk("JUNIPER HEALTH", "thousands", "R&D expense", 2023, 9870, 8510),
    mk("KAPPA LOGISTICS", "thousands", "Total current assets", 2023, 33120, 29880),
    mk("LUMEN MEDIA", "thousands", "Interest expense", 2023, -2150, -1980),
    mk("MERIDIAN FIN", "thousands", "Noninterest expense", 2023, 18600, 17400),
    mk("NIMBUS CLOUD", "thousands", "Cash from operations", 2023, 7430, 5900),
    mk("ORION AEROSPACE", "millions", "Gross profit", 2023, 3410, 3080),
    mk("PINNACLE REALTY", "millions", "Total liabilities", 2023, 27800, 26900),
    mk("QUASAR BIOTECH", "millions", "Net loss", 2023, -1205, -990),
    mk("RIGEL MOTORS", "millions", "Cash and equivalents", 2023, 6090, 5400),
    mk("SOLSTICE APPAREL", "thousands", "Revenue", 2022, 43900, 40110),
    mk("TITAN STEEL", "thousands", "Net income", 2022, 3480, 2760),
    mk("UMBRA ANALYTICS", "thousands", "Total revenue", 2023, 512340, 448900),
    mk("VEGA UTILITIES", "millions", "Total assets", 2023, 103700, 98400),
    mk("WILLOW DESIGN", "thousands", "Operating income (loss)", 2022, 1240, -890),
]


def main() -> None:
    # A strong target (gpt-4o) on purpose — the point is that even a strong model
    # needs the convention. The baseline names the task but not the scale/sign/format.
    agent = fa.Agent(
        name="financials-extractor",
        system_prompt=(
            "You extract values from financial statements. "
            "Answer the question with the requested value."
        ),
        llm=fa.LLMClient(provider="openai", model="gpt-4o"),
    )

    print("=== AutoLLM — improving financial-statement extraction (real OpenAI) ===\n")
    report = fa.optimize(
        agent,
        DATASET,
        [NumericMatch()],  # grade the number (magnitude + sign), not the punctuation
        config=fa.OptimizeConfig(
            levers=("instructions",),
            max_iterations=8,
            candidates_per_iteration=4,
            patience=4,
            seed=0,
        ),
        # A strong proposer: it must read the failures (with expected values) and
        # articulate the scale/sign convention.
        proposer_llm=fa.LLMClient(provider="openai", model="gpt-4o"),
        run_name="autollm financials demo",
        persist=True,  # land the run in local.db for `fastaiagent ui`
    )

    print(report.summary())
    print()
    print(f"baseline dev = {report.baseline.score:.3f}    best dev = {report.best.score:.3f}")
    print("→ improved" if report.improved else "→ no improvement this run")

    print("\n=== the prompt you started with ===")
    print(agent.system_prompt)
    print("\n=== the prompt your evals found ===")
    print(report.best_candidate.system_prompt or "(unchanged from baseline)")

    tuned = report.apply_to(agent)
    probe = (
        "Q: Operating income (loss) for 2022?\n"
        "APEX ROBOTICS (in millions)       2023        2022\n"
        "Operating income (loss)   240   (155)"
    )
    print("\n=== tuned agent on a held-out probe (in millions, parentheses → negative) ===")
    print(probe)
    print(f"answer : {tuned.run(probe).output!r}    (expected -155000000)")

    if report.run_id:
        print(
            f"\nRun persisted as {report.run_id[:8]} — open `fastaiagent ui` → AutoLLM."
        )


if __name__ == "__main__":
    main()
