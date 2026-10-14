import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { Route, Routes } from "react-router-dom";
import { renderWithProviders } from "@/test/utils";
import { SimulationDetailPage } from "./SimulationDetailPage";
import { SimulationsPage } from "./SimulationsPage";
import type { SimulationCase, SimulationRunRow } from "@/lib/types";

const RUN: SimulationRunRow = {
  run_id: "s1",
  run_name: "refund flows",
  agent_name: "support",
  scenario_count: 4,
  pass_count: 1,
  fail_count: 3,
  pass_rate: 0.25,
  started_at: "2026-10-14T10:00:00+00:00",
  finished_at: null,
  metadata: {},
  errored_count: 2,
  judged_count: 2,
};

function scenario(id: string, over: Partial<SimulationCase> = {}): SimulationCase {
  return {
    case_id: id,
    run_id: "s1",
    ordinal: 0,
    scenario_name: id,
    passed: 0,
    criteria: { success: [], failure: [] },
    per_criterion: [],
    transcript: [],
    trace_id: null,
    error: null,
    ...over,
  };
}

/** api.ts boundary (fetch) is the only mock surface. */
function mockSim(cases: SimulationCase[]) {
  return vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
    const url = new URL(String(input), "http://localhost");
    const body = url.pathname === "/api/simulations"
      ? { rows: [RUN], total: 1, page: 1, page_size: 100 }
      : { run: RUN, cases, total_cases: 4 };
    return Promise.resolve(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    );
  });
}

function stat(label: string): HTMLElement {
  return document.querySelector<HTMLElement>(`[data-stat="${label}"]`)!;
}

describe("SimulationDetailPage", () => {
  it("shows passed, failed (judged) and errored apart, and a neutral ERRORED badge", async () => {
    mockSim([
      scenario("timeout", { error: "agent timed out after 3 turns" }),
      scenario("judge-down", {
        error: "judge could not run on 1 criteria: polite",
        per_criterion: [
          { criterion: "polite", kind: "success", passed: false, reason: "503", errored: true },
        ],
      }),
      scenario("rude", { per_criterion: [{ criterion: "polite", kind: "success", passed: false, reason: "no" }] }),
    ]);
    renderWithProviders(
      <Routes>
        <Route path="/simulations/:runId" element={<SimulationDetailPage />} />
      </Routes>,
      { route: "/simulations/s1" }
    );

    const badge = await screen.findByTestId("scenario-badge-timeout");
    expect(badge).toHaveTextContent("ERRORED");
    expect(badge.className).not.toMatch(/destructive/);
    expect(screen.getByText("agent timed out after 3 turns")).toBeInTheDocument();
    expect(screen.getByTestId("scenario-badge-rude")).toHaveTextContent("FAIL");

    expect(stat("Passed")).toHaveTextContent("1");
    expect(stat("Failed")).toHaveTextContent("1");
    expect(stat("Errored")).toHaveTextContent("2");

    // An errored criterion reads "could not run", in neutral.
    fireEvent.click(screen.getByText("judge-down"));
    const chip = screen.getByText("· could not run").parentElement!;
    expect(chip).toHaveAttribute("data-errored", "true");
    expect(chip.className).not.toMatch(/destructive/);
  });

  it("shows the hardening recommendations recorded on the run", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            run: {
              ...RUN,
              metadata: {
                hardening: {
                  model: "gpt-4o",
                  failure_count: 1,
                  recommendations: [
                    { target: "tools", recommendation: "Add a refund lookup tool", rationale: "It guessed" },
                  ],
                },
              },
            },
            cases: [],
            total_cases: 0,
          }),
          { status: 200, headers: { "content-type": "application/json" } }
        )
      )
    );
    renderWithProviders(
      <Routes>
        <Route path="/simulations/:runId" element={<SimulationDetailPage />} />
      </Routes>,
      { route: "/simulations/s1" }
    );
    const card = await screen.findByTestId("hardening-card");
    expect(card).toHaveTextContent("Add a refund lookup tool");
    expect(card).toHaveTextContent("1 failure analysed");
  });

  it("filters by the errored outcome", async () => {
    const spy = mockSim([scenario("timeout", { error: "boom" })]);
    renderWithProviders(
      <Routes>
        <Route path="/simulations/:runId" element={<SimulationDetailPage />} />
      </Routes>,
      { route: "/simulations/s1" }
    );
    await screen.findByTestId("scenario-badge-timeout");
    fireEvent.change(screen.getByLabelText("Outcome"), { target: { value: "errored" } });
    await waitFor(() =>
      expect(String(spy.mock.calls.at(-1)![0])).toContain("outcome=errored")
    );
  });
});

describe("SimulationsPage", () => {
  it("shows each run's errored count", async () => {
    mockSim([]);
    renderWithProviders(<SimulationsPage />);
    const cell = await screen.findByTestId("sim-errored-s1");
    expect(within(cell).getByText("2")).toBeInTheDocument();
    // The fraction is over all scenarios, like the rate, whatever fail_count holds.
    expect(screen.getByText("(1/4)")).toBeInTheDocument();
  });
});
