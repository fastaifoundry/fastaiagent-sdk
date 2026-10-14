import { describe, expect, it } from "vitest";
import { screen, within } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { AgentCiPanel } from "./AgentCiPanel";

describe("AgentCiPanel", () => {
  it("renders with only an outcome (every other part missing)", () => {
    renderWithProviders(<AgentCiPanel gate={{ outcome: "invalid" }} runId="r2" />);
    const panel = screen.getByTestId("agent-ci-panel");
    expect(within(panel).getByText("invalid")).toHaveAttribute("data-gate", "invalid");
    expect(within(panel).getByText("No threshold was demanded.")).toBeInTheDocument();
    expect(screen.queryByTestId("gate-baseline")).toBeNull();
    expect(screen.queryByTestId("gate-error-bound")).toBeNull();
  });

  it("tolerates null parts", () => {
    renderWithProviders(
      <AgentCiPanel
        gate={{
          outcome: null,
          thresholds: null,
          checks: null,
          error: null,
          invalid_reasons: null,
          baseline: null,
          git_sha: null,
          git_branch: null,
        }}
        runId="r2"
      />
    );
    expect(screen.getByTestId("agent-ci-panel")).toBeInTheDocument();
  });

  it("shows thresholds with actual values, the error bound, reasons, baseline and commit", () => {
    renderWithProviders(
      <AgentCiPanel
        runId="r2"
        gate={{
          outcome: "failed",
          thresholds: { "overall.pass_rate": 0.9 },
          checks: [
            { metric: "overall", field: "pass_rate", minimum: 0.9, actual: 0.8, passed: false },
            { metric: "judge", field: "avg_score", minimum: 0.5, actual: null, passed: false },
          ],
          error: { errored: 1, scored: 9, error_rate: 0.1, max_error_rate: 0.2 },
          invalid_reasons: ["judge: the scorer could not run on 1 case(s)"],
          baseline: {
            run_id: "base-1",
            run_name: "main",
            pass_rate_delta: -0.05,
            case_pass_rate_delta: -0.1,
            regressed_count: 2,
            pass_rate_comparable: false,
          },
          git_sha: "abcdef1234567",
          git_branch: "feature/x",
        }}
      />
    );
    const table = screen.getByTestId("gate-thresholds");
    expect(table).toHaveTextContent("overall.pass_rate");
    expect(table).toHaveTextContent("≥ 0.9");
    expect(table).toHaveTextContent("0.8000");
    expect(within(table).getByText("no value")).toBeInTheDocument();
    expect(screen.getByTestId("gate-error-bound")).toHaveTextContent(
      "1 errored of 10 cases (10%) · allowed ≤ 20%"
    );
    expect(screen.getByText("judge: the scorer could not run on 1 case(s)")).toBeInTheDocument();
    const baseline = screen.getByTestId("gate-baseline");
    expect(within(baseline).getByRole("link", { name: "main" })).toHaveAttribute(
      "href",
      "/evals/base-1"
    );
    expect(within(baseline).getByRole("link", { name: "Compare" })).toHaveAttribute(
      "href",
      "/evals/compare?a=base-1&b=r2"
    );
    expect(baseline).toHaveTextContent("cases Δ -10% · checks Δ -5% · 2 regressed");
    expect(baseline).toHaveTextContent(/not like for like/);
    expect(screen.getByText("abcdef1")).toHaveAttribute("title", "abcdef1234567");
    expect(screen.getByText("feature/x")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy commit SHA" })).toBeInTheDocument();
  });
});
