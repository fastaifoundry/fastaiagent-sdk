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
