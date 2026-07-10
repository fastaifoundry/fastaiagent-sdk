"""AutoLLM on REAL annual reports — bank-compliance extraction across companies.

The task a credit team actually has: read a company's jaarrekening and pull the
figures + covenant ratios a bank needs. This example uses **three real, downloadable
Dutch annual reports** (2023), and every figure below was transcribed from them and
reconciles (equity + long-term + short-term = balance sheet total):

    Newtone  GCF IV TopCo 9 B.V.   full euros
             https://newtone.nl/wp-content/uploads/2024/09/Jaarrekening-2023_Newtone.pdf
    NS       Nederlandse Spoorwegen  € millions
             https://www.nsjaarverslag.nl/jaarverslag-2023/jaarrekening/geconsolideerde-jaarrekening/geconsolideerde-balans-per-31-december-2023
    Enexis   Enexis Holding N.V.     € millions
             https://publicaties.enexis.nl/jaarverslag/jaarverslag-2023/verslag/jaarrekening-2023/geconsolideerde-balans

We use the SDK to run a `gpt-5.4-mini` agent over compact excerpts (never the whole
report — real TPM budgets don't allow dumping) and extract compliance attributes as
JSON, then let **AutoLLM** (`fastaiagent.optimize`) improve the extraction prompt.

**Why three companies, not one.** A single document lets the optimizer *cheat* —
it can memorise that company's answers ("solvabiliteit = 35.4") and still pass a
held-out split drawn from the same doc. Across three very different companies,
memorising one company's numbers fails on the others, so the only way to raise the
score is to recover the *transferable* conventions a bank actually needs:

  - **Scale** — Newtone is in euros; NS and Enexis are "in miljoenen euro's" (×1,000,000).
  - **Label mapping** — *Groepsvermogen* / *Totaal eigen vermogen* → equity; which
    result line is the net result; subtotal vs. sub-line.
  - **Sign** — losses print with a minus / in parentheses.
  - **The bank's ratio definitions** — solvabiliteit = eigen vermogen / balanstotaal;
    nettoschuld = langlopende schulden − liquide middelen; debt/equity uses BOTH
    long- and short-term debt; EBITDA = bedrijfsresultaat + afschrijvingen; etc.,
    with fixed rounding — the same formulas for every company.

Grading is a per-field numeric scorer (partial credit). Ratios are computed here in
Python from the real figures, so the gold is always internally consistent.

Target : gpt-5.4-mini.   Proposer : gpt-5.4.   Real OpenAI calls, no mocks.
**Requires FastAIAgent >= 1.38.0** (the proposer must see expected values to recover
these conventions).

Run
---
    export OPENAI_API_KEY=sk-...
    python examples/autollm/jaarrekening.py
    fastaiagent ui            # open -> AutoLLM -> the "autollm jaarrekening demo" run
"""

from __future__ import annotations

import json
import re
from typing import Any

import fastaiagent as fa
from fastaiagent.eval.scorer import Scorer, ScorerResult

# ── Real figures per company (native reporting unit) ─────────────────────────
# factor = multiplier to full euros (1 for Newtone, 1_000_000 for the "in miljoenen").
COMPANIES: list[dict[str, Any]] = [
    {
        "id": "Newtone",
        "entity": "GCF IV TopCo 9 B.V. (Newtone) — Geconsolideerde jaarrekening 2023",
        "url": "https://newtone.nl/wp-content/uploads/2024/09/Jaarrekening-2023_Newtone.pdf",
        "unit": "euro's",
        "factor": 1,
        "f": dict(
            netto_omzet=72292322, bedrijfsresultaat=-760658, resultaat_na_belastingen=-5805550,
            afschrijvingen=11547156, betaalde_interest=-4595417, balanstotaal=219947714,
            eigen_vermogen=77918006, langlopende_schulden=111724957, kortlopende_schulden=29452531,
            vaste_activa=170762443, vlottende_activa=49185271, liquide_middelen=14401876,
        ),
    },
    {
        "id": "NS",
        "entity": "Nederlandse Spoorwegen (NS Groep N.V.) — Geconsolideerde jaarrekening 2023",
        "url": "https://www.nsjaarverslag.nl/jaarverslag-2023/jaarrekening/geconsolideerde-jaarrekening/geconsolideerde-balans-per-31-december-2023",
        "unit": "miljoenen euro's",
        "factor": 1_000_000,
        # NS's net financing result is a small income (+43), so there is no clean
        # interest-expense line — interest-coverage attributes are omitted for NS.
        "f": dict(
            netto_omzet=3763, bedrijfsresultaat=-540, resultaat_na_belastingen=-380,
            afschrijvingen=995, betaalde_interest=None, balanstotaal=6375,
            eigen_vermogen=1914, langlopende_schulden=2263, kortlopende_schulden=2198,
            vaste_activa=4507, vlottende_activa=1868, liquide_middelen=460,
        ),
    },
    {
        "id": "Enexis",
        "entity": "Enexis Holding N.V. — Geconsolideerde jaarrekening 2023",
        "url": "https://publicaties.enexis.nl/jaarverslag/jaarverslag-2023/verslag/jaarrekening-2023/geconsolideerde-balans",
        "unit": "miljoenen euro's",
        "factor": 1_000_000,
        "f": dict(
            netto_omzet=2014, bedrijfsresultaat=109, resultaat_na_belastingen=72,
            afschrijvingen=468, betaalde_interest=-21, balanstotaal=10460,
            eigen_vermogen=5320, langlopende_schulden=4679, kortlopende_schulden=461,
            vaste_activa=9916, vlottende_activa=544, liquide_middelen=127,
        ),
    },
    {
        "id": "Liander",
        "entity": "Liander N.V. — Geconsolideerde jaarrekening 2023",
        "url": "https://www.liander.nl/-/media/files/financiele-communicatie/jaarverslagen/liander_jaarbericht_2023.pdf",
        "unit": "miljoenen euro's",
        "factor": 1_000_000,
        # Liander holds no material own cash — it settles via a rekening-courant with
        # its parent Alliander, so there is no separate 'liquide middelen' balance-sheet
        # line (liquide_middelen = 0).
        "f": dict(
            netto_omzet=2510, bedrijfsresultaat=456, resultaat_na_belastingen=273,
            afschrijvingen=444, betaalde_interest=-93, balanstotaal=10307,
            eigen_vermogen=3036, langlopende_schulden=4555, kortlopende_schulden=2716,
            vaste_activa=9703, vlottende_activa=604, liquide_middelen=0,
        ),
    },
    {
        "id": "Stedin",
        "entity": "Stedin Netbeheer B.V. — Geconsolideerde jaarrekening 2023",
        "url": "https://www.stedin.net/-/media/project/online/files/jaarverslagen-en-publicaties/jaarbericht-2023-stedin-netbeheer-bv.pdf",
        "unit": "miljoenen euro's",
        "factor": 1_000_000,
        "f": dict(
            netto_omzet=1716, bedrijfsresultaat=280, resultaat_na_belastingen=195,
            afschrijvingen=305, betaalde_interest=-16, balanstotaal=8959,
            eigen_vermogen=4493, langlopende_schulden=2541, kortlopende_schulden=1925,
            vaste_activa=8327, vlottende_activa=632, liquide_middelen=59,
        ),
    },
]


def _nl(v: float) -> str:
    """Render a value the Dutch way: '.' as thousands separator, '()' for negatives."""
    n = int(round(v))
    s = f"{abs(n):,}".replace(",", ".")
    return f"({s})" if n < 0 else s


def derive(f: dict[str, Any]) -> dict[str, float]:
    """Compute the bank's compliance ratios from the raw figures (native unit).

    Ratios/percentages are scale-invariant; absolute derived values (ebitda,
    nettoschuld, werkkapitaal) are returned in the native unit and scaled later.
    """
    ebitda = f["bedrijfsresultaat"] + f["afschrijvingen"]
    # Net debt (standard): all debt minus cash.
    nettoschuld = f["langlopende_schulden"] + f["kortlopende_schulden"] - f["liquide_middelen"]
    vv = f["balanstotaal"] - f["eigen_vermogen"]  # vreemd vermogen
    d = {
        "ebitda": ebitda,
        "nettoschuld": nettoschuld,
        "werkkapitaal": f["vlottende_activa"] - f["kortlopende_schulden"],
        "ebitda_marge_pct": round(ebitda / f["netto_omzet"] * 100, 1),
        "solvabiliteit_pct": round(f["eigen_vermogen"] / f["balanstotaal"] * 100, 1),
        "schuldratio_pct": round(vv / f["balanstotaal"] * 100, 1),
        "current_ratio": round(f["vlottende_activa"] / f["kortlopende_schulden"], 2),
        "cash_ratio": round(f["liquide_middelen"] / f["kortlopende_schulden"], 2),
        "debt_to_equity": round(
            (f["langlopende_schulden"] + f["kortlopende_schulden"]) / f["eigen_vermogen"], 2),
        "gearing": round(nettoschuld / f["eigen_vermogen"], 2),
        "nettoschuld_ebitda": round(nettoschuld / ebitda, 2),
        "roe_pct": round(f["resultaat_na_belastingen"] / f["eigen_vermogen"] * 100, 1),
        "roa_pct": round(f["resultaat_na_belastingen"] / f["balanstotaal"] * 100, 1),
        "activa_omloopsnelheid": round(f["netto_omzet"] / f["balanstotaal"], 2),
        "kapitaalintensiteit_pct": round(f["vaste_activa"] / f["balanstotaal"] * 100, 1),
    }
    if f.get("betaalde_interest"):
        d["rentedekkingsgraad"] = round(ebitda / abs(f["betaalde_interest"]), 2)
    return d


# Which attributes are absolute euro amounts (must be scaled to full euros) vs
# scale-invariant ratios/percentages.
_RATIO_KEYS = {
    "ebitda_marge_pct", "solvabiliteit_pct", "schuldratio_pct", "current_ratio",
    "cash_ratio", "debt_to_equity", "gearing", "nettoschuld_ebitda", "roe_pct",
    "roa_pct", "activa_omloopsnelheid", "kapitaalintensiteit_pct", "rentedekkingsgraad",
}


def build_dataset() -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for co in COMPANIES:
        f, factor, d = co["f"], co["factor"], derive(co["f"])
        # A compact "kerncijfers" excerpt in the company's native unit + provenance.
        lines = [
            f"[Bron: {co['url']} — {co['entity']}]",
            f"Kerncijfers 2023 (geconsolideerd; bedragen in {co['unit']})",
            f"Netto-omzet                    {_nl(f['netto_omzet'])}",
            f"Bedrijfsresultaat              {_nl(f['bedrijfsresultaat'])}",
            f"Resultaat na belastingen       {_nl(f['resultaat_na_belastingen'])}",
            f"Afschrijvingen                 {_nl(f['afschrijvingen'])}",
        ]
        if f.get("betaalde_interest"):
            lines.append(f"Betaalde interest              {_nl(f['betaalde_interest'])}")
        lines += [
            f"Balanstotaal                   {_nl(f['balanstotaal'])}",
            f"Eigen vermogen (Groepsvermogen){_nl(f['eigen_vermogen']):>15}",
            f"Langlopende schulden           {_nl(f['langlopende_schulden'])}",
            f"Kortlopende schulden           {_nl(f['kortlopende_schulden'])}",
            f"Vaste activa                   {_nl(f['vaste_activa'])}",
            f"Vlottende activa               {_nl(f['vlottende_activa'])}",
            f"Liquide middelen               {_nl(f['liquide_middelen'])}",
        ]
        kern = "\n".join(lines)

        def gold(keys: list[str]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for k in keys:
                if k in d:
                    out[k] = d[k] if k in _RATIO_KEYS else int(d[k] * factor)
                else:
                    out[k] = int(f[k] * factor)
            return out

        def case(keys: list[str]) -> None:
            g = {k: v for k, v in gold(keys).items()}
            prompt = (
                f"{kern}\n\n"
                f"Gevraagde posten (voor kredietbeoordeling): {', '.join(keys)}.\n"
                "Geef uitsluitend een JSON-object met exact deze sleutels."
            )
            cases.append({"input": prompt, "expected_output": json.dumps(g)})

        # ~5 themed cases per company (banks track many covenant ratios).
        case(["netto_omzet", "bedrijfsresultaat", "resultaat_na_belastingen",
              "ebitda", "ebitda_marge_pct"])
        case(["vlottende_activa", "kortlopende_schulden", "current_ratio",
              "cash_ratio", "werkkapitaal"])
        case(["eigen_vermogen", "balanstotaal", "solvabiliteit_pct",
              "debt_to_equity", "schuldratio_pct"])
        lev = ["nettoschuld", "nettoschuld_ebitda", "gearing"]
        if "rentedekkingsgraad" in d:
            lev.append("rentedekkingsgraad")
        case(lev)
        case(["roe_pct", "roa_pct", "activa_omloopsnelheid", "kapitaalintensiteit_pct"])
        # a comprehensive ~15-attribute credit profile (figures + covenant ratios)
        case(["netto_omzet", "ebitda", "resultaat_na_belastingen", "balanstotaal",
              "eigen_vermogen", "langlopende_schulden", "kortlopende_schulden",
              "liquide_middelen", "nettoschuld", "werkkapitaal", "solvabiliteit_pct",
              "current_ratio", "debt_to_equity", "nettoschuld_ebitda", "gearing",
              "roe_pct"])
    return cases


DATASET = build_dataset()


class ComplianceFields(Scorer):
    """Per-field numeric match over the requested attributes (partial credit).

    Absolute euro amounts must match exactly; ratios/percentages (|value| < 1000)
    allow a small absolute tolerance for rounding. Score = fraction correct; the
    reason lists every miss so the proposer can see exactly what to fix.
    """

    name = "compliance_fields"

    @staticmethod
    def _num(x: Any) -> float | None:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace("%", "").replace(" ", "")
        m = re.search(r"-?\d[\d.,]*", s)
        if not m:
            return None
        tok = m.group(0)
        neg = tok.startswith("-")
        tok = tok.lstrip("-")
        if "," in tok:
            tok = tok.replace(".", "").replace(",", ".")
        elif re.match(r"^\d{1,3}(\.\d{3})+$", tok):
            tok = tok.replace(".", "")
        try:
            v = float(tok)
        except ValueError:
            return None
        return -v if neg else v

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        t = re.sub(r"^```(?:json)?|```$", "", str(text).strip(), flags=re.MULTILINE).strip()
        m = re.search(r"\{.*\}", t, re.DOTALL)
        try:
            return json.loads(m.group(0) if m else t)
        except Exception:
            return {}

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        gold = json.loads(expected) if expected else {}
        got = self._parse_json(output)
        if not gold:
            return ScorerResult(score=0.0, passed=False, reason="no expected fields")
        correct, misses = 0, []
        for key, exp_val in gold.items():
            exp = self._num(exp_val)
            val = self._num(got.get(key)) if isinstance(got, dict) else None
            tol = 0.1 if abs(exp) < 1000 else max(1.0, abs(exp) * 1e-6)
            if val is not None and abs(val - exp) <= tol:
                correct += 1
            else:
                seen = got.get(key) if isinstance(got, dict) else None
                misses.append(f"{key}: got {seen}, expected {exp_val}")
        return ScorerResult(
            score=correct / len(gold),
            passed=(correct == len(gold)),
            reason=None if not misses else "; ".join(misses),
        )


# A deliberately LARGE, professional baseline prompt — a realistic credit-analysis
# SOP. It reads thoroughly, but it never pins the things that actually bite on Dutch
# statements: the unit scaling ("report in euros … exactly as presented" is
# self-contradictory for an "in miljoenen" statement), the label synonyms, the sign
# convention, subtotal-vs-line, the bank's *exact* ratio formulas, and the rounding.
# Those gaps are the headroom AutoLLM recovers — transferably, across companies.
BASELINE_PROMPT = (
    "You are a senior financial-analysis assistant embedded in a bank's credit-risk "
    "and compliance function. Your job is to read excerpts from the annual accounts "
    "(jaarrekeningen) of corporate borrowers and produce a clean, structured set of "
    "the financial data points the credit team needs to assess the obligor. You will "
    "receive a compact excerpt from a company's consolidated financial statements "
    "(winst-en-verliesrekening, balans, and/or kerncijfers) together with a list of "
    "requested attributes.\n\n"
    "Work methodically and conservatively, as a careful analyst would:\n"
    "1. Read the entire excerpt before answering, and identify where each requested "
    "attribute appears or from which figures it must be derived.\n"
    "2. Match each requested attribute to the correct line item. Be precise: do not "
    "confuse similarly named lines (for example operating result versus result after "
    "tax, or a subtotal versus one of its components).\n"
    "3. For monetary line items, report the amount for the requested reporting year "
    "exactly as presented in the statement, in euros.\n"
    "4. Where a requested attribute is a standard financial ratio or KPI rather than a "
    "reported line, compute it from the figures available in the excerpt using "
    "generally accepted definitions.\n"
    "5. Present results in a single JSON object whose keys are exactly the requested "
    "attribute names, with numeric values. If a value is genuinely not present and "
    "cannot be derived, use null rather than guessing.\n\n"
    "Be professional and consistent. Do not include explanations, units, currency "
    "symbols, or commentary — only the JSON object. Double-check that every requested "
    "key is present and that you have not added extra keys. Accuracy and consistency "
    "across companies matter more than speed: the same attribute must be computed the "
    "same way for every borrower."
)


def main() -> None:
    agent = fa.Agent(
        name="jaarrekening-extractor",
        system_prompt=BASELINE_PROMPT,
        llm=fa.LLMClient(provider="openai", model="gpt-5.4-mini"),
    )

    print("=== AutoLLM — bank-compliance extraction across 3 real jaarrekeningen ===")
    print(f"    {len(DATASET)} cases from {len(COMPANIES)} companies "
          f"({', '.join(c['id'] for c in COMPANIES)})\n")
    report = fa.optimize(
        agent,
        DATASET,
        [ComplianceFields()],
        config=fa.OptimizeConfig(
            levers=("instructions",),
            max_iterations=8,
            candidates_per_iteration=4,
            patience=4,
            seed=0,
            primary_metric="compliance_fields",
        ),
        proposer_llm=fa.LLMClient(provider="openai", model="gpt-5.4"),
        run_name="autollm jaarrekening demo",
        persist=True,
    )

    print(report.summary())
    print()
    print(f"baseline field-accuracy = {report.baseline.score:.3f}  "
          f"best = {report.best.score:.3f}")
    print("→ improved" if report.improved else "→ no improvement this run")

    print("\n=== the (large) prompt you started with ===")
    print(BASELINE_PROMPT)
    print("\n=== the prompt your evals found (transferable, across companies) ===")
    print(report.best_candidate.system_prompt or "(unchanged from baseline)")

    if report.run_id:
        print(f"\nRun persisted as {report.run_id[:8]} — open `fastaiagent ui` → AutoLLM.")


if __name__ == "__main__":
    main()
