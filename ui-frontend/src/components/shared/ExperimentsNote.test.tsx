import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { ExperimentsNote, IncludeExperimentsToggle } from "./ExperimentsNote";
import { OverviewPage } from "@/pages/OverviewPage";
import { AgentsPage } from "@/pages/AgentsPage";
import { usePreferencesStore } from "@/stores/preferences-store";

/**
 * Production views leave experiments out by default, and say what they left
 * out — cost included — so excluding them never hides spend (1.89.0). The
 * api.ts boundary (fetch) is the only mock surface.
 */

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

beforeEach(() => {
  usePreferencesStore.setState({ includeExperiments: false });
});

afterEach(() => vi.restoreAllMocks());

describe("ExperimentsNote", () => {
  it("names the excluded traces, their cost and their sources", () => {
    renderWithProviders(
      <ExperimentsNote
        count={12}
        costUsd={0.42}
        bySource={{
          playground: { trace_count: 2 },
          replay: { trace_count: 10 },
        }}
        included={false}
      />
    );
    const note = screen.getByTestId("experiments-note");
    // Largest source first; the cost is shown even though it is excluded.
    expect(note).toHaveTextContent(
      "Excludes 12 experiment traces ($0.4200: replay 10, playground 2)"
    );
  });

  it("shows a zero cost rather than hiding the cost", () => {
    renderWithProviders(
      <ExperimentsNote count={1} costUsd={0} bySource={{ eval: { trace_count: 1 } }} included={false} />
    );
    expect(screen.getByTestId("experiments-note")).toHaveTextContent(
      "Excludes 1 experiment trace ($0.00: eval 1)"
    );
  });

  it("says Includes when the numbers already count them", () => {
    renderWithProviders(<ExperimentsNote count={3} included />);
    expect(screen.getByTestId("experiments-note")).toHaveTextContent(
      /^Includes 3 experiment traces\.$/
    );
  });

  it("renders nothing when there are no experiment traces", () => {
    renderWithProviders(<ExperimentsNote count={0} costUsd={0} included={false} />);
    expect(screen.queryByTestId("experiments-note")).not.toBeInTheDocument();
  });

  it("the toggle flips the persisted preference", () => {
    renderWithProviders(<IncludeExperimentsToggle />);
    const toggle = screen.getByRole("switch", { name: "Include experiments" });
    expect(toggle).toHaveAttribute("aria-checked", "false");
    fireEvent.click(toggle);
    expect(usePreferencesStore.getState().includeExperiments).toBe(true);
    expect(toggle).toHaveAttribute("aria-checked", "true");
  });
});

describe("OverviewPage experiments", () => {
  function stubOverview() {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      const included = url.includes("include_experiments=true");
      if (url.includes("/api/overview")) {
        return json({
          traces_last_24h: included ? 7 : 4,
          experiments_last_24h: 3,
          experiments_included: included,
          failing_traces_last_24h: 0,
          eval_runs_last_7d: 0,
          avg_pass_rate_last_7d: 0,
          pending_approvals_count: 0,
          failed_executions_count: 0,
          recent_traces: [],
          recent_eval_runs: [],
          prompt_changes_last_7d: [],
          agents_with_errors: [],
        });
      }
      return json({});
    });
  }

  function overviewUrls(): string[] {
    return vi
      .mocked(globalThis.fetch)
      .mock.calls.map(([u]) => String(u))
      .filter((u) => u.includes("/api/overview"));
  }

  it("asks without experiments by default and says how many it left out", async () => {
    stubOverview();
    renderWithProviders(<OverviewPage />);
    const note = await screen.findByTestId("experiments-note");
    expect(note).toHaveTextContent("Excludes 3 experiment traces from the last 24h");
    expect(overviewUrls()[0]).toContain("include_experiments=false");
  });

  it("refetches with include_experiments=true when the toggle is on", async () => {
    stubOverview();
    renderWithProviders(<OverviewPage />);
    await screen.findByTestId("experiments-note");
    fireEvent.click(screen.getByRole("switch", { name: "Include experiments" }));
    await waitFor(() =>
      expect(screen.getByTestId("experiments-note")).toHaveTextContent(
        "Includes 3 experiment traces"
      )
    );
    expect(overviewUrls().some((u) => u.includes("include_experiments=true"))).toBe(true);
  });
});

describe("AgentsPage experiments", () => {
  function stubAgents() {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if (url.includes("/api/agents")) {
        const included = url.includes("include_experiments=true");
        return json({
          agents: [
            {
              agent_name: "support-bot",
              run_count: included ? 9 : 4,
              success_rate: 1,
              error_count: 0,
              avg_latency_ms: 120,
              avg_cost_usd: 0.001,
              last_run: new Date().toISOString(),
            },
          ],
        });
      }
      return json({});
    });
  }

  function agentsUrls(): string[] {
    return vi
      .mocked(globalThis.fetch)
      .mock.calls.map(([u]) => String(u))
      .filter((u) => u.includes("/api/agents"));
  }

  it("asks without experiments by default and says the counts are production-only", async () => {
    stubAgents();
    renderWithProviders(<AgentsPage />);
    await screen.findByText("support-bot");
    expect(agentsUrls()[0]).toContain("include_experiments=false");
    expect(screen.getByTestId("production-only-note")).toHaveTextContent(
      "Counts production runs only; replay reruns and Playground calls are excluded."
    );
  });

  it("refetches with include_experiments=true and drops the note when toggled on", async () => {
    stubAgents();
    renderWithProviders(<AgentsPage />);
    await screen.findByText("support-bot");
    fireEvent.click(screen.getByRole("switch", { name: "Include experiments" }));
    await waitFor(() =>
      expect(agentsUrls().some((u) => u.includes("include_experiments=true"))).toBe(true)
    );
    await waitFor(() => expect(screen.getByText("9")).toBeInTheDocument());
    expect(screen.queryByTestId("production-only-note")).not.toBeInTheDocument();
  });
});
