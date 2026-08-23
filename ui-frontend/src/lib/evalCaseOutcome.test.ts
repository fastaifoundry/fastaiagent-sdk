import { describe, expect, it } from "vitest";

import { evalCaseOutcome } from "./types";

/**
 * Mirrors the backend's `_case_outcome` (fastaiagent/eval/compare.py).
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

  it("treats a scored-but-empty case as passed (matches the backend)", () => {
    expect(evalCaseOutcome({ error: null, per_scorer: {} })).toBe("passed");
  });
});
