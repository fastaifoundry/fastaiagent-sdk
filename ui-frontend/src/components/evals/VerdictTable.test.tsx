import { describe, expect, it } from "vitest";
import { fireEvent, screen, within } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { VerdictTable } from "./VerdictTable";

describe("VerdictTable", () => {
  it("shows errored as a neutral 'could not run' and n/a as n/a, with no score for either", () => {
    renderWithProviders(
      <VerdictTable
        perScorer={{
          exact_match: { passed: true, score: 1 },
          judge: { passed: false, score: 0, errored: true, reason: "Judge error: 503" },
          trajectory: { passed: false, score: 0, skipped: true, reason: "no expected trajectory" },
        }}
      />
    );
    const judgeRow = screen.getByTestId("verdict-row-judge");
    const badge = within(judgeRow).getByText("could not run");
    expect(badge).toHaveAttribute("data-outcome", "errored");
    expect(badge.className).not.toMatch(/destructive/);
    expect(screen.getByTestId("verdict-score-judge")).toHaveTextContent("—");
    expect(within(judgeRow).getByText("Judge error: 503")).toBeInTheDocument();

    const trajRow = screen.getByTestId("verdict-row-trajectory");
    expect(within(trajRow).getByText("n/a")).toHaveAttribute("data-outcome", "skipped");
    expect(screen.getByTestId("verdict-score-trajectory")).toHaveTextContent("—");
    // The placeholder 0 is never shown as a score.
    expect(screen.queryByText("0.00")).toBeNull();

    expect(screen.getByTestId("verdict-score-exact_match")).toHaveTextContent("1.00");
    expect(within(screen.getByTestId("verdict-row-exact_match")).getByText("passed")).toBeInTheDocument();
  });

  it("shows a judge's raw score on its scale and its threshold", () => {
    renderWithProviders(
      <VerdictTable
        perScorer={{
          helpful: {
            passed: true,
            score: 0.5,
            details: { raw_score: 3, scale: "1-5", threshold: 0.5 },
          },
          binary: {
            passed: false,
            score: 0,
            details: { raw_score: 0, scale: "binary", threshold: 0.5 },
          },
        }}
      />
    );
    expect(screen.getByTestId("verdict-score-helpful")).toHaveTextContent("3/5 → 0.50 (≥0.50)");
    expect(screen.getByTestId("verdict-score-binary")).toHaveTextContent("0.00 (≥0.50)");
  });

  it("lists a trajectory scorer's expected and actual tools", () => {
    renderWithProviders(
      <VerdictTable
        perScorer={{
          tool_order: {
            passed: false,
            score: 0,
            details: { expected: ["search", "answer"], actual: ["answer"] },
          },
        }}
      />
    );
    expect(screen.getByText("expected: search → answer")).toBeInTheDocument();
    expect(screen.getByText("actual: answer")).toBeInTheDocument();
  });

  it("folds a long reason and expands it on demand", () => {
    const long = "x".repeat(350) + "END";
    renderWithProviders(
      <VerdictTable perScorer={{ judge: { passed: false, score: 0.1, reason: long } }} />
    );
    expect(screen.queryByText(/END/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Show more" }));
    expect(screen.getByText(/END/)).toBeInTheDocument();
  });

  it("links each verdict to the case's scoring trace", () => {
    renderWithProviders(
      <VerdictTable
        perScorer={{ exact_match: { passed: true, score: 1 } }}
        scoringTraceId="score-trace-1"
      />
    );
    expect(screen.getByRole("link", { name: /scoring/ })).toHaveAttribute(
      "href",
      "/traces/score-trace-1"
    );
  });
});
