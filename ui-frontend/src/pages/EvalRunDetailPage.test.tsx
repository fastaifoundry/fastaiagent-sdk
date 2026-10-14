import { describe, expect, it, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import { Route, Routes } from "react-router-dom";
import { renderWithProviders } from "@/test/utils";
import { EvalRunDetailPage } from "./EvalRunDetailPage";
import type { EvalCaseRow, EvalRunDetail, EvalRunRow } from "@/lib/types";

function run(over: Partial<EvalRunRow> = {}): EvalRunRow {
  return {
    run_id: "r1",
    run_name: "nightly",
    dataset_name: "qa.jsonl",
    agent_name: "support",
    agent_version: null,
    scorers: ["exact_match", "judge"],
    started_at: "2026-10-14T10:00:00+00:00",
    finished_at: "2026-10-14T10:01:05+00:00",
    pass_count: 16,
    fail_count: 4,
    pass_rate: 0.8,
    metadata: {},
    kind: "evaluate",
    summary_version: 2,
    case_count: 10,
    case_pass_count: 6,
    case_fail_count: 2,
    case_error_count: 2,
    case_pass_rate: 0.75,
    check_error_count: 1,
    check_skip_count: 4,
    agent_cost_usd: 0.0123,
    agent_cost_known: true,
    eval_cost_usd: 0.004,
    eval_cost_known: false,
    cost_usd: 0.0123,
    avg_latency_ms: 2100,
    scorer_summary: {
      exact_match: { pass: 7, fail: 2, errored: 0, skipped: 1 },
      judge: { pass: 9, fail: 2, errored: 1, skipped: 3 },
    },
    gate: null,
    ...over,
  };
}

const CASE: EvalCaseRow = {
  case_id: "c1",
  run_id: "r1",
  ordinal: 0,
  input: "hi",
  expected_output: "hello",
  actual_output: "hello",
  trace_id: "t1",
  per_scorer: { exact_match: { passed: true, score: 1 } },
  error: null,
  scoring_trace_id: "st1",
};

/** api.ts boundary (fetch) is the only mock surface. */
function mockRun(detail: EvalRunDetail) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(() =>
    Promise.resolve(
      new Response(JSON.stringify(detail), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    )
  );
}

function renderPage() {
  return renderWithProviders(
    <Routes>
      <Route path="/evals/:runId" element={<EvalRunDetailPage />} />
    </Routes>,
    { route: "/evals/r1" }
  );
}

function stat(label: string): HTMLElement {
  const el = document.querySelector<HTMLElement>(`[data-stat="${label}"]`);
  if (!el) throw new Error(`no stat ${label}`);
  return el;
}

describe("EvalRunDetailPage headline", () => {
  it("shows cases first, then checks, from the run row (not the filtered list)", async () => {
    // Only one case is in the (filtered) list; the run row says 2 errored.
    mockRun({ run: run(), cases: [CASE], total_cases: 10 });
    renderPage();
    await screen.findByTestId("eval-headline");

    expect(stat("Cases passed")).toHaveTextContent("75%");
    expect(stat("Cases passed")).toHaveTextContent("6/8");
    expect(stat("Failed")).toHaveTextContent("2");
    expect(stat("Errored")).toHaveTextContent("2");
    expect(stat("Checks")).toHaveTextContent("80%");
    expect(stat("Checks")).toHaveTextContent("16/20 · 4 n/a · 1 errored");
    expect(stat("Checks").getAttribute("title")).toMatch(/--eval-fail-under/);
    expect(stat("Agent cost")).toHaveTextContent("$0.0123");
    expect(stat("Eval cost")).toHaveTextContent("partial");
    expect(stat("Started")).toHaveTextContent("took 1m 5s");
    // The case-list chip also reads the run row.
    expect(screen.getByText("2 errored")).toBeInTheDocument();
  });

  it("renders — for a null rate and an untraced cost, never 0%", async () => {
    mockRun({
      run: run({
        case_pass_rate: null,
        pass_rate: null,
        case_pass_count: 0,
        case_fail_count: 0,
        case_error_count: 3,
        agent_cost_usd: null,
        agent_cost_known: false,
        eval_cost_usd: null,
        eval_cost_known: false,
      }),
      cases: [],
      total_cases: 3,
    });
    renderPage();
    await screen.findByTestId("eval-headline");
    expect(within(stat("Cases passed")).getByText("—")).toBeInTheDocument();
    expect(within(stat("Checks")).getByText("—")).toBeInTheDocument();
    expect(stat("Cases passed")).not.toHaveTextContent("0%");
    const agentValue = within(stat("Agent cost")).getByText("—");
    expect(agentValue.getAttribute("title")).toMatch(/not traced/);
    expect(within(stat("Eval cost")).getByText("—")).toBeInTheDocument();
  });

  it("shows scorer chips with errored and n/a counts", async () => {
    mockRun({ run: run(), cases: [CASE], total_cases: 10 });
    renderPage();
    const chips = await screen.findByTestId("scorer-chips");
    expect(chips).toHaveTextContent("judge9/11· 1 errored · 3 n/a");
    expect(chips).toHaveTextContent("exact_match7/9· 1 n/a");
  });

  it("explains a dataset check and a run stored under the old rules", async () => {
    mockRun({
      run: run({ kind: "ui-echo", agent_name: "editor-echo", summary_version: null }),
      cases: [CASE],
      total_cases: 1,
    });
    renderPage();
    expect(await screen.findByTestId("echo-banner")).toHaveTextContent(
      "Dataset check — not a model evaluation."
    );
    expect(screen.getByTestId("legacy-banner")).toHaveTextContent(/stored before 1.89/);
  });

  it("shows no banners for a current evaluate run", async () => {
    mockRun({ run: run(), cases: [CASE], total_cases: 1 });
    renderPage();
    await screen.findByTestId("eval-headline");
    expect(screen.queryByTestId("echo-banner")).toBeNull();
    expect(screen.queryByTestId("legacy-banner")).toBeNull();
    expect(screen.queryByTestId("agent-ci-panel")).toBeNull();
  });

  it("shows the gate panel, scorer configuration and hardening when recorded", async () => {
    mockRun({
      run: run({
        gate: { outcome: "failed", checks: [], thresholds: { "overall.pass_rate": 0.9 } },
        metadata: {
          scorer_config: {
            judge: { type: "LLMJudge", criteria: "helpfulness", scale: "1-5", model: "gpt-4o" },
          },
          hardening: {
            model: "gpt-4o",
            failure_count: 2,
            recommendations: [
              { target: "system_prompt", recommendation: "Ask for the order id", rationale: "Two cases lacked it" },
            ],
          },
        },
      }),
      cases: [CASE],
      total_cases: 1,
    });
    renderPage();
    const panel = await screen.findByTestId("agent-ci-panel");
    expect(within(panel).getByText("failed")).toBeInTheDocument();
    const config = screen.getByTestId("scorer-config-card");
    expect(config).toHaveTextContent("helpfulness");
    expect(config).toHaveTextContent("LLMJudge");
    const hardening = screen.getByTestId("hardening-card");
    expect(hardening).toHaveTextContent("Ask for the order id");
    expect(hardening).toHaveTextContent("2 failures analysed");
  });
});
