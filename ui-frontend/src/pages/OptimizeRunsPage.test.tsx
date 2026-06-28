import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { OptimizeRunsPage } from "./OptimizeRunsPage";
import type { OptimizeRunRow } from "@/lib/types";

function row(overrides: Partial<OptimizeRunRow>): OptimizeRunRow {
  return {
    run_id: "r",
    run_name: null,
    agent_name: null,
    baseline_score: 0.5,
    best_score: 0.8,
    holdout_baseline_score: null,
    holdout_best_score: null,
    reverted: 0,
    stopped_reason: "target_score",
    seed: 0,
    levers: ["instructions"],
    config: {},
    best_candidate: {},
    baseline_eval_run_id: null,
    best_eval_run_id: null,
    iteration_count: 2,
    started_at: new Date().toISOString(),
    finished_at: new Date().toISOString(),
    metadata: {},
    ...overrides,
  };
}

function mockApi(rows: OptimizeRunRow[]) {
  global.fetch = vi.fn(() =>
    Promise.resolve(
      new Response(JSON.stringify({ rows, total: rows.length, page: 1, page_size: 100 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    )
  ) as unknown as typeof fetch;
}

describe("OptimizeRunsPage", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("shows an empty state when nothing has run", async () => {
    mockApi([]);
    renderWithProviders(<OptimizeRunsPage />);
    await waitFor(() =>
      expect(screen.getByText(/No AutoLLM runs yet/i)).toBeInTheDocument()
    );
  });

  it("lists runs from multiple agents and offers an agent filter", async () => {
    mockApi([
      row({ run_id: "a1", run_name: "kyc-onboarding", agent_name: "kyc" }),
      row({ run_id: "b1", run_name: "refund-bot", agent_name: "refund" }),
    ]);
    renderWithProviders(<OptimizeRunsPage />);

    await waitFor(() =>
      expect(screen.getByText("kyc-onboarding")).toBeInTheDocument()
    );
    // Each run carries its agent tag (the multi-agent case is one row per run).
    expect(screen.getByText("kyc")).toBeInTheDocument();
    expect(screen.getByText("refund")).toBeInTheDocument();
    // The agent filter appears once more than one agent is present.
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("hides the agent filter when only one agent has runs", async () => {
    mockApi([row({ run_id: "a1", run_name: "kyc-onboarding", agent_name: "kyc" })]);
    renderWithProviders(<OptimizeRunsPage />);
    await waitFor(() =>
      expect(screen.getByText("kyc-onboarding")).toBeInTheDocument()
    );
    expect(screen.queryByRole("combobox")).toBeNull();
  });
});
