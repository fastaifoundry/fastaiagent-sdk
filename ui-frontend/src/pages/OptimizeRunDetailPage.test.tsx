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

  it("shows the winning prompt next to the one the run started from", async () => {
    const detail = fixture();
    detail.run.metadata = { baseline_system_prompt: "You answer questions." };
    mockFetch(detail);
    renderDetail();

    await waitFor(() => expect(screen.getByText("Winning prompt")).toBeInTheDocument());
    expect(screen.getByText("better")).toBeInTheDocument();
    expect(screen.getByText("Original prompt")).toBeInTheDocument();
    expect(screen.getByText("You answer questions.")).toBeInTheDocument();
    expect(screen.getByTitle("Copy winning prompt")).toBeInTheDocument();
  });

  it("says a reverted run keeps the original configuration", async () => {
    const detail = fixture();
    detail.run.reverted = 1;
    detail.run.best_candidate = { system_prompt: null, fewshot_demos: null, fact_ids: null };
    detail.run.metadata = { baseline_system_prompt: "You answer questions." };
    mockFetch(detail);
    renderDetail();

    await waitFor(() =>
      expect(screen.getByText(/regressed on the holdout and was reverted/i)).toBeInTheDocument()
    );
    expect(screen.queryByText("Winning prompt")).not.toBeInTheDocument();
    expect(screen.getByText("Prompt (unchanged)")).toBeInTheDocument();
  });

  it("lists the few-shot examples and learned facts a winner selected", async () => {
    const detail = fixture();
    detail.run.best_candidate = {
      system_prompt: null,
      fewshot_demos: [
        { input: "Capital of France?", output: "Paris" },
        { input: "Capital of Japan?", output: "Tokyo" },
      ],
      fact_ids: [3, 7],
    };
    mockFetch(detail);
    renderDetail();

    await waitFor(() => expect(screen.getByText("2 few-shot examples")).toBeInTheDocument());
    expect(screen.getByText("Capital of France?")).toBeInTheDocument();
    expect(screen.getByText("Learned facts injected: #3, #7")).toBeInTheDocument();
    expect(screen.getByText("System prompt: unchanged.")).toBeInTheDocument();
  });

  it("surfaces a proposer that could not run", async () => {
    const detail = fixture();
    detail.run.stopped_reason = "proposer_failed";
    detail.run.best_candidate = { system_prompt: null };
    detail.run.metadata = {
      proposer_errors: ["LLMProviderError: model_not_found", "LLMProviderError: model_not_found"],
    };
    mockFetch(detail);
    renderDetail();

    await waitFor(() =>
      expect(screen.getByText(/The prompt proposer failed 2 times/)).toBeInTheDocument()
    );
    expect(screen.getByText(/No candidate beat the baseline/)).toBeInTheDocument();
  });
});
