import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { EvalComparePage } from "./EvalComparePage";

/**
 * Smoke test only — asserts the page imports and mounts without throwing
 * when no ?a=&b= is in the URL. The real compare flow is exercised by:
 *   - tests/test_ui_evals_enhanced.py (real SQLite, real FastAPI)
 *   - ui-frontend/tests/screenshots.spec.ts (live server, seeded fixtures)
 * per the repo's no-mocking-of-subject-under-test rule.
 */
describe("EvalComparePage", () => {
  it("renders the 'Pick two runs' empty state when no runs are chosen", () => {
    renderWithProviders(<EvalComparePage />, { route: "/evals/compare" });
    expect(screen.getByText("Pick two runs")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /Compare eval runs/i })
    ).toBeInTheDocument();
  });
});
