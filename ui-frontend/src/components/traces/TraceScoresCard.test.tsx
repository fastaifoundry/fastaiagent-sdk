import { describe, expect, it, vi } from "vitest";
import { waitFor, screen } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { TraceScoresCard } from "./TraceScoresCard";

describe("TraceScoresCard", () => {
  it("renders nothing when the trace has no scores", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          trace_id: "t-1",
          guardrail_events: [],
          eval_cases: [],
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      )
    );

    const { container } = renderWithProviders(<TraceScoresCard traceId="t-1" />);

    await waitFor(() => expect((fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.length).toBeGreaterThan(0));
    await waitFor(() => {
      expect(container.querySelector("[data-slot='card']")).toBeNull();
    });
  });

  it("renders guardrail + eval chips when scores exist", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          trace_id: "t-2",
          guardrail_events: [
            {
              event_id: "e1",
              guardrail_name: "no_pii",
              outcome: "blocked",
              score: 0,
              message: "SSN detected",
              timestamp: new Date().toISOString(),
            },
          ],
          eval_cases: [
            {
              case_id: "c1",
              run_id: "r1",
              ordinal: 0,
              per_scorer: {
                exact_match: { passed: true, score: 1.0 },
              },
              run_name: "smoke",
              dataset_name: "d1",
              started_at: new Date().toISOString(),
              input: "hi",
              expected_output: "hi",
              actual_output: "hi",
            },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      )
    );

    renderWithProviders(<TraceScoresCard traceId="t-2" />);

    await screen.findByText("no_pii");
    expect(screen.getByText("blocked")).toBeInTheDocument();
    expect(screen.getByText(/exact_match pass/)).toBeInTheDocument();
    expect(screen.getByText(/in smoke/)).toBeInTheDocument();
  });

  it("explains a scoring trace and a simulated conversation, errored in neutral", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          trace_id: "t-4",
          guardrail_events: [],
          eval_cases: [],
          eval_scoring: [
            {
              case_id: "c7",
              run_id: "run-77",
              ordinal: 3,
              per_scorer: {
                exact_match: { passed: true, score: 1 },
                judge: { passed: false, score: 0, errored: true, reason: "Judge error" },
                trajectory: { passed: false, score: 0, skipped: true },
              },
              trace_id: "agent-trace-7",
              run_name: "nightly",
              dataset_name: "qa",
              started_at: new Date().toISOString(),
            },
          ],
          simulation_cases: [
            {
              case_id: "s9",
              run_id: "sim-9",
              ordinal: 0,
              scenario_name: "refund",
              passed: 0,
              per_criterion: [],
              error: "judge could not run on 1 criteria: polite",
              run_name: "flows",
              started_at: new Date().toISOString(),
            },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      )
    );

    renderWithProviders(<TraceScoresCard traceId="t-4" />);

    const scoring = await screen.findByRole("link", { name: "Scoring for case #3 of run nightly" });
    expect(scoring).toHaveAttribute("href", "/evals/run-77");
    expect(screen.getByRole("link", { name: /agent trace/ })).toHaveAttribute(
      "href",
      "/traces/agent-trace-7"
    );
    const judge = screen.getByText(/judge could not run/);
    expect(judge).toHaveAttribute("data-outcome", "errored");
    expect(judge.className).not.toMatch(/destructive/);
    expect(judge).not.toHaveTextContent("0.00");
    expect(screen.getByText(/trajectory n\/a/)).toHaveAttribute("data-outcome", "skipped");

    expect(
      screen.getByRole("link", { name: "Simulated conversation: scenario refund of run flows" })
    ).toHaveAttribute("href", "/simulations/sim-9");
    const sim = screen.getByTestId("sim-outcome-s9");
    expect(sim).toHaveTextContent("could not run");
    expect(sim.className).not.toMatch(/destructive/);
  });

  it("shows errored as an amber could-not-run, maps reasked, and adds the effect", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          trace_id: "t-3",
          guardrail_events: [
            {
              event_id: "e1",
              guardrail_name: "judge",
              outcome: "errored",
              score: null,
              message: "LLM call failed",
              timestamp: new Date().toISOString(),
              effect: "unchecked",
            },
            {
              event_id: "e2",
              guardrail_name: "on_topic",
              outcome: "reasked",
              score: null,
              message: null,
              timestamp: new Date().toISOString(),
            },
          ],
          eval_cases: [],
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      )
    );

    renderWithProviders(<TraceScoresCard traceId="t-3" />);

    await screen.findByText("judge");
    const errored = screen.getByText("could not run");
    expect(errored).toHaveAttribute("data-tone", "warning");
    expect(errored.className).toMatch(/fa-warning/);
    expect(errored.className).not.toMatch(/destructive/);
    expect(screen.getByText("· unchecked")).toBeInTheDocument();
    expect(screen.getByText("re-asked")).toHaveAttribute("data-tone", "warning");
  });
});
