import { describe, expect, it, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { OverviewPage } from "./OverviewPage";

const BASE = {
  traces_last_24h: 4,
  experiments_last_24h: 0,
  experiments_included: false,
  failing_traces_last_24h: 0,
  eval_runs_last_7d: 5,
  avg_pass_rate_last_7d: 0.1,
  eval_case_pass_rate_7d: null as number | null,
  eval_datasets_7d: 0,
  eval_runs_counted_7d: 3,
  eval_runs_excluded_7d: 2,
  eval_gate_failures_7d: 0,
  pending_approvals_count: 0,
  failed_executions_count: 0,
  recent_traces: [],
  recent_eval_runs: [] as unknown[],
  prompt_changes_last_7d: [],
  agents_with_errors: [],
};

/** api.ts boundary (fetch) is the only mock surface. */
function mockOverview(over: Partial<typeof BASE>) {
  vi.spyOn(globalThis, "fetch").mockImplementation(() =>
    Promise.resolve(
      new Response(JSON.stringify({ ...BASE, ...over }), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    )
  );
}

function stat(label: string): HTMLElement {
  return document.querySelector<HTMLElement>(`[data-stat="${label}"]`)!;
}

describe("OverviewPage evals", () => {
  it("headlines the cases rate, — when null, and says what was counted", async () => {
    mockOverview({});
    renderWithProviders(<OverviewPage />);
    await screen.findByText("3 counted, 2 left out", { exact: false });
    const tile = stat("Eval cases passed (7d)");
    expect(within(tile).getByText("—")).toBeInTheDocument();
    // The legacy checks average (10%) is not the headline any more.
    expect(tile).not.toHaveTextContent("10%");
    expect(screen.queryByText(/pass rate is/)).toBeNull();
  });

  it("raises the gate-failure attention item, linking to failed gated runs", async () => {
    mockOverview({ eval_case_pass_rate_7d: 0.9, eval_datasets_7d: 1, eval_gate_failures_7d: 2 });
    renderWithProviders(<OverviewPage />);
    expect(await screen.findByText("2 gated eval runs failed or were invalid")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /View gated runs/ })).toHaveAttribute(
      "href",
      "/evals?gate=failed,invalid"
    );
    expect(stat("Eval cases passed (7d)")).toHaveTextContent("90%");
  });

  it("chips recent runs with their cases rate and gate", async () => {
    mockOverview({
      recent_eval_runs: [
        {
          run_id: "r1",
          run_name: "nightly",
          dataset_name: "qa",
          pass_rate: 0.8,
          started_at: new Date().toISOString(),
          kind: "pytest",
          case_pass_rate: null,
          gate_outcome: "invalid",
        },
      ],
    });
    renderWithProviders(<OverviewPage />);
    const link = await screen.findByRole("link", { name: "nightly" });
    const li = link.closest("li")!;
    expect(within(li).getByText("invalid")).toBeInTheDocument();
    expect(within(li).getByText("—")).toHaveAttribute("title", "Cases — · Checks 80%");
    expect(within(li).getByText("pytest")).toBeInTheDocument();
  });
});
