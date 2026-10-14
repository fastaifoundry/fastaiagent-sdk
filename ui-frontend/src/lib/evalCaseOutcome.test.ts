import { describe, expect, it } from "vitest";

import { evalCaseOutcome, evalVerdictOutcome } from "./types";

/**
 * Mirrors the backend's `case_outcome_detail` (fastaiagent/eval/compare.py).
 *
 * The regression this guards: an errored case carries an EMPTY `per_scorer`
 * by design, and `every()` over an empty array is `true` — so before the
 * `error` check existed, an infrastructure failure rendered as "passed" and
 * an outage looked like a perfect run.
 */
describe("evalCaseOutcome", () => {
  it("classifies an infra failure as errored, not passed", () => {
    expect(evalCaseOutcome({ error: "provider 503", per_scorer: {} })).toBe("errored");
  });

  it("errored wins even if scorer verdicts somehow exist", () => {
    expect(
      evalCaseOutcome({
        error: "timeout",
        per_scorer: { exact_match: { passed: true, score: 1 } },
      }),
    ).toBe("errored");
  });

  it("passes when every scorer passed", () => {
    expect(
      evalCaseOutcome({
        error: null,
        per_scorer: {
          exact_match: { passed: true, score: 1 },
          faithfulness: { passed: true, score: 0.9 },
        },
      }),
    ).toBe("passed");
  });

  it("fails when any scorer failed", () => {
    expect(
      evalCaseOutcome({
        error: null,
        per_scorer: {
          exact_match: { passed: true, score: 1 },
          faithfulness: { passed: false, score: 0.2 },
        },
      }),
    ).toBe("failed");
  });

  // 1.89.0: a case nothing checked used to read as a pass.
  it("treats a case with no verdict as errored (unscored)", () => {
    expect(evalCaseOutcome({ error: null, per_scorer: {} })).toBe("errored");
  });

  it("treats a case whose every check was n/a as errored (unscored)", () => {
    expect(
      evalCaseOutcome({
        error: null,
        per_scorer: {
          exact_match: { passed: false, score: 0, skipped: true },
          trajectory: { passed: false, score: 0, skipped: true },
        },
      }),
    ).toBe("errored");
  });

  it("errors when a check could not run and none failed", () => {
    expect(
      evalCaseOutcome({
        error: null,
        per_scorer: {
          exact_match: { passed: true, score: 1 },
          judge: { passed: false, score: 0, errored: true },
        },
      }),
    ).toBe("errored");
  });

  it("a fail stays a fail even when another check errored", () => {
    expect(
      evalCaseOutcome({
        error: null,
        per_scorer: {
          exact_match: { passed: false, score: 0 },
          judge: { passed: false, score: 0, errored: true },
        },
      }),
    ).toBe("failed");
  });

  it("passes when one check passed and the rest were n/a", () => {
    expect(
      evalCaseOutcome({
        error: null,
        per_scorer: {
          exact_match: { passed: true, score: 1 },
          trajectory: { passed: false, score: 0, skipped: true },
        },
      }),
    ).toBe("passed");
  });
});

describe("evalVerdictOutcome", () => {
  it("reads errored and skipped before passed", () => {
    expect(evalVerdictOutcome({ passed: false, score: 0, errored: true })).toBe("errored");
    expect(evalVerdictOutcome({ passed: false, score: 0, skipped: true })).toBe("skipped");
    expect(evalVerdictOutcome({ passed: true, score: 1 })).toBe("passed");
    expect(evalVerdictOutcome({ passed: false, score: 0.2 })).toBe("failed");
  });
});
