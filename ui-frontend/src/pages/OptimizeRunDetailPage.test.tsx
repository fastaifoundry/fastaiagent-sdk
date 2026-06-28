import { describe, expect, it, vi } from "vitest";
import { Route, Routes } from "react-router-dom";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { OptimizeRunDetailPage } from "./OptimizeRunDetailPage";
import type { OptimizeRunDetail } from "@/lib/types";

/** api.ts boundary (fetch) is the only mock surface, per the testing decision. */
function mockFetch(payload: unknown) {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "content-type": "application/json" },
    })
  );
}

function fixture(): OptimizeRunDetail {
  return {
    run: {
      run_id: "run-1",
      run_name: "kyc-opt",
      agent_name: "kyc",
      baseline_score: 0.5,
      best_score: 0.8,
      holdout_baseline_score: 0.5,
      holdout_best_score: 0.7,
      reverted: 0,
      stopped_reason: "target_score",
      seed: 7,
      levers: ["instructions"],
      config: {},
      best_candidate: { system_prompt: "better" },
      baseline_eval_run_id: "ev-0",
      best_eval_run_id: "ev-1",
      iteration_count: 2,
      started_at: new Date().toISOString(),
      finished_at: new Date().toISOString(),
      metadata: {},
    },
    iterations: [
      {
        iteration_id: "it-0",
        run_id: "run-1",
        ordinal: 0,
        iteration: 0,
        lever: "baseline",
        candidate_id: "c0",
        dev_score: 0.5,
        accepted: 1,
        skipped: 0,
        rationale: "baseline",
        eval_run_id: "ev-0",
      },
      {
        iteration_id: "it-1",
        run_id: "run-1",
        ordinal: 1,
        iteration: 1,
        lever: "instructions",
        candidate_id: "c1",
        dev_score: 0.8,
        accepted: 1,
        skipped: 0,
        rationale: "clearer instructions",
        eval_run_id: "ev-1",
      },
    ],
    total_iterations: 2,
  };
}

function renderDetail() {
  return renderWithProviders(
    <Routes>
      <Route path="/optimizes/:runId" element={<OptimizeRunDetailPage />} />
    </Routes>,
    { route: "/optimizes/run-1" }
  );
}

describe("OptimizeRunDetailPage", () => {
  it("renders the trajectory with lever attribution and an eval drill-down link", async () => {
    mockFetch(fixture());
    renderDetail();

    await waitFor(() =>
      expect(screen.getByText("clearer instructions")).toBeInTheDocument()
    );
    // Lever attribution: the per-iteration rationale renders.
    expect(screen.getByText("clearer instructions")).toBeInTheDocument();

    // Each iteration drills into the eval run that scored its candidate.
    const links = screen.getAllByTitle(
      /Open the eval run that scored this candidate/i
    );
    expect(links).toHaveLength(2);
    expect(links[1]).toHaveAttribute("href", "/evals/ev-1");
  });
});
