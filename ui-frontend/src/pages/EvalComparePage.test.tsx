import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen, within } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { EvalComparePage } from "./EvalComparePage";
import type { EvalCaseRow, EvalCompareResponse, EvalRunRow } from "@/lib/types";

/**
 * The api.ts boundary (fetch) is the only mock surface; the compare logic is
 * exercised for real by tests/test_ui_evals_enhanced.py (real SQLite, real
 * FastAPI).
 */
describe("EvalComparePage", () => {
  it("renders the 'Pick two runs' empty state when no runs are chosen", () => {
    renderWithProviders(<EvalComparePage />, { route: "/evals/compare" });
    expect(screen.getByText("Pick two runs")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /Compare eval runs/i })
    ).toBeInTheDocument();
  });

  function run(id: string, over: Partial<EvalRunRow> = {}): EvalRunRow {
    return {
      run_id: id,
      run_name: id,
      dataset_name: "qa",
      agent_name: "support",
      agent_version: null,
      scorers: ["exact_match"],
      started_at: "2026-10-14T10:00:00+00:00",
      finished_at: null,
      pass_count: 3,
      fail_count: 1,
      pass_rate: 0.75,
      metadata: {},
      summary_version: 2,
      case_pass_count: 3,
      case_fail_count: 1,
      case_error_count: 0,
      case_pass_rate: 0.75,
      agent_cost_usd: null,
      agent_cost_known: false,
      eval_cost_usd: null,
      eval_cost_known: false,
      ...over,
    };
  }

  function kase(id: string, ordinal: number, over: Partial<EvalCaseRow> = {}): EvalCaseRow {
    return {
      case_id: id,
      run_id: "r",
      ordinal,
      input: `input ${ordinal}`,
      expected_output: "same",
      actual_output: "same",
      trace_id: null,
      per_scorer: { exact_match: { passed: true, score: 1 } },
      error: null,
      ...over,
    };
  }

  function mockCompare(body: Partial<EvalCompareResponse>) {
    const full: EvalCompareResponse = {
      run_a: run("base"),
      run_b: run("head"),
      regressed: [],
      improved: [],
      unchanged_pass: 1,
      unchanged_fail: 0,
      pass_rate_delta: 0,
      cost_delta_usd: 0,
      errored: [],
      only_in_a: [],
      only_in_b: [],
      drifted: [],
      case_pass_rate_delta: 0,
      pass_rate_comparable: true,
      agent_cost_delta_usd: null,
      eval_cost_delta_usd: null,
      ...body,
    };
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      const payload = url.includes("/evals/compare")
        ? full
        : { rows: [full.run_a, full.run_b], total: 2, page: 1, page_size: 100 };
      return Promise.resolve(
        new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "content-type": "application/json" },
        })
      );
    });
  }

  it("lists errored, unmatched and drifted cases in collapsed sections", async () => {
    mockCompare({
      case_pass_rate_delta: -0.25,
      pass_rate_delta: -0.1,
      errored: [
        {
          a: kase("a1", 1),
          b: kase("b1", 1, { error: "provider 503", per_scorer: {} }),
          scorer_deltas: [],
          side: "b",
        },
      ],
      only_in_a: [kase("a9", 9, { input: "dropped case" })],
      only_in_b: [kase("b7", 7, { input: "new case" })],
      drifted: [
        {
          a: kase("a2", 2),
          b: kase("b2", 2),
          scorer_deltas: [
            {
              scorer: "judge",
              passed_before: true,
              passed_after: true,
              changed: false,
              score_before: 0.9,
              score_after: 0.6,
              score_delta: -0.3,
            },
          ],
        },
      ],
    });
    renderWithProviders(<EvalComparePage />, { route: "/evals/compare?a=base&b=head" });

    const errored = await screen.findByTestId("compare-errored");
    expect(screen.getByText("Cases Δ").parentElement?.parentElement).toHaveTextContent("-25%");
    expect(screen.getByTestId("compare-summary")).toHaveTextContent(
      "1 errored, 2 unmatched, 1 with score drift"
    );

    // Collapsed until opened.
    expect(within(errored).queryByText(/provider 503/)).toBeNull();
    fireEvent.click(within(errored).getByRole("button", { name: /Errored — not compared \(1\)/ }));
    expect(within(errored).getByText("errored in B (after)")).toBeInTheDocument();
    expect(within(errored).getByText(/provider 503/)).toBeInTheDocument();

    const unmatched = screen.getByTestId("compare-unmatched");
    fireEvent.click(within(unmatched).getByRole("button", { name: /Unmatched/ }));
    expect(within(unmatched).getByText("Only in A (baseline) · 1")).toBeInTheDocument();
    expect(within(unmatched).getByText("dropped case")).toBeInTheDocument();
    expect(within(unmatched).getByText("Only in B (current) · 1")).toBeInTheDocument();
    expect(within(unmatched).getByText("new case")).toBeInTheDocument();

    const drift = screen.getByTestId("compare-drift");
    fireEvent.click(within(drift).getByRole("button", { name: /Score drift/ }));
    expect(drift).toHaveTextContent("judge: 0.90 → 0.60 (-0.30)");

    // No regressed / improved: the empty state points at the sections above.
    expect(screen.getByText("No case changed between passed and failed")).toBeInTheDocument();
    expect(screen.queryByTestId("not-comparable-warning")).toBeNull();
  });

  it("warns when the checks rates were stored under different rules", async () => {
    mockCompare({ pass_rate_comparable: false, pass_rate_delta: 0.2 });
    renderWithProviders(<EvalComparePage />, { route: "/evals/compare?a=base&b=head" });
    expect(await screen.findByTestId("not-comparable-warning")).toHaveTextContent(
      /not like for like/
    );
    expect(screen.queryByTestId("compare-errored")).toBeNull();
    expect(
      screen.getByText(/No case errored, every case matched, and no score moved/)
    ).toBeInTheDocument();
  });

  it("renders — for a null checks delta, never 0%", async () => {
    mockCompare({ pass_rate_delta: null, case_pass_rate_delta: null });
    renderWithProviders(<EvalComparePage />, { route: "/evals/compare?a=base&b=head" });
    await screen.findByTestId("compare-summary");
    const checks = document.querySelector<HTMLElement>('[data-stat="Checks Δ"]')!;
    const cases = document.querySelector<HTMLElement>('[data-stat="Cases Δ"]')!;
    expect(within(checks).getByText("—")).toHaveAttribute("title", "a run has no checks rate");
    expect(checks).not.toHaveTextContent("0%");
    expect(within(cases).getByText("—")).toBeInTheDocument();
    expect(screen.queryByText(/net improvement|net regression/)).toBeNull();
  });

  it("loads each run's thumbnails from its own case and compares images by hash", async () => {
    const img = (sha: string, idx: number) => ({
      type: "image",
      media_type: "image/png",
      size_bytes: 2048,
      sha256: sha,
      media_idx: idx,
    });
    mockCompare({
      regressed: [
        {
          a: kase("case-a", 4, { run_id: "base", actual_output: img("aaa111", 0) }),
          b: kase("case-b", 4, {
            run_id: "head",
            // Same image, another media_idx: still identical.
            expected_output: img("bbb222", 0),
            actual_output: img("bbb222", 1),
            per_scorer: { exact_match: { passed: false, score: 0 } },
          }),
          scorer_deltas: [
            { scorer: "exact_match", passed_before: true, passed_after: false, changed: true },
          ],
        },
      ],
    });
    renderWithProviders(<EvalComparePage />, { route: "/evals/compare?a=base&b=head" });
    const panels = await screen.findAllByTestId("media-compare");
    expect(panels).toHaveLength(2);
    expect(panels[0]).toHaveTextContent("Expected and Actual (B) · Identical");
    expect(panels[1]).toHaveTextContent("Actual (B) and Actual (A) · Differ");
    const srcs = within(panels[1])
      .getAllByRole("img")
      .map((i) => i.getAttribute("src"));
    expect(srcs).toEqual([
      "/api/evals/head/cases/case-b/media/1",
      "/api/evals/base/cases/case-a/media/0",
    ]);
    // Never the descriptor JSON.
    expect(panels[1]).not.toHaveTextContent("media_idx");
  });

  it("shows an Identical panel instead of an empty diff when expected equals actual", async () => {
    mockCompare({
      improved: [
        {
          a: kase("a3", 3, { actual_output: "wrong", per_scorer: { exact_match: { passed: false, score: 0 } } }),
          b: kase("b3", 3),
          scorer_deltas: [
            { scorer: "exact_match", passed_before: false, passed_after: true, changed: true },
          ],
        },
      ],
    });
    renderWithProviders(<EvalComparePage />, { route: "/evals/compare?a=base&b=head" });
    const panel = await screen.findByTestId("identical-panel");
    expect(panel).toHaveTextContent("Expected and Actual (B) · Identical");
    expect(screen.queryByText(/@@ -0,1 \+0,1 @@/)).toBeNull();
  });
});
