import { describe, expect, it, vi } from "vitest";
import { Route, Routes } from "react-router-dom";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { GuardrailEventDetailPage } from "./GuardrailEventDetailPage";
import type { GuardrailEvent, GuardrailEventDetail } from "@/lib/types";

/** api.ts boundary (fetch) is the only mock surface, per the testing decision. */
function mockFetch(payload: unknown) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(() =>
    Promise.resolve(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    )
  );
}

function event(overrides: Partial<GuardrailEvent> = {}): GuardrailEvent {
  return {
    event_id: "e1",
    trace_id: "t1",
    span_id: "s1",
    guardrail_name: "no_pii",
    guardrail_type: "pii",
    position: "output",
    outcome: "blocked",
    score: null,
    message: "PII found",
    agent_name: "support",
    timestamp: "2026-10-14T10:00:00.000002+00:00",
    metadata: {},
    false_positive: false,
    false_positive_at: null,
    action: "block",
    action_taken: "blocked",
    severity: "high",
    floor: true,
    effect: "blocked",
    final_reason: null,
    latency_ms: 12,
    attempt: null,
    batch_id: "b1",
    guardrail_span_id: "gs1",
    judged_ref: "ref",
    rule: null,
    blocking: true,
    on_error: "block",
    origin: "plane",
    ...overrides,
  };
}

function detail(
  ev: GuardrailEvent,
  overrides: Partial<GuardrailEventDetail> = {}
): GuardrailEventDetail {
  return {
    event: ev,
    judged: { text: "My SSN is 123-45-6789", recorded: true },
    trigger: {
      kind: "agent_output",
      text: "parent span text",
      content_type: "agent output",
      span_name: "agent.support",
    },
    context: { spans: [], sibling_events: [] },
    ...overrides,
  };
}

function renderDetail() {
  return renderWithProviders(
    <Routes>
      <Route path="/guardrail-events/:eventId" element={<GuardrailEventDetailPage />} />
    </Routes>,
    { route: "/guardrail-events/e1" }
  );
}

describe("GuardrailEventDetailPage", () => {
  it("shows exactly the judged text in Panel 1, not the parent span", async () => {
    mockFetch(detail(event()));
    renderDetail();

    const panel = await screen.findByTestId("panel-trigger");
    expect(within(panel).getByText("My SSN is 123-45-6789")).toBeInTheDocument();
    expect(within(panel).queryByText("parent span text")).not.toBeInTheDocument();
    expect(screen.queryByText(/FASTAIAGENT_TRACE_PAYLOADS/)).not.toBeInTheDocument();
  });

  it("falls back to the span text for a legacy row, and says when nothing was recorded", async () => {
    const legacy = event({ effect: null, judged_ref: null });
    mockFetch(
      detail(legacy, {
        judged: { text: null, recorded: false },
        trigger: { kind: "agent_output", text: null, content_type: "agent output" },
      })
    );
    renderDetail();

    const panel = await screen.findByTestId("panel-trigger");
    expect(
      within(panel).getByText("Not recorded — this event predates 1.89.0.")
    ).toBeInTheDocument();
    // The old hint was wrong: payload capture does not govern guardrail rows.
    expect(screen.queryByText(/FASTAIAGENT_TRACE_PAYLOADS/)).not.toBeInTheDocument();
  });

  it("says when an earlier rule in the same pass rewrote the judged text", async () => {
    const sibling = event({
      event_id: "e0",
      guardrail_name: "mask_emails",
      outcome: "filtered",
      effect: "rewritten",
      timestamp: "2026-10-14T10:00:00.000001+00:00",
    });
    mockFetch(detail(event(), { context: { spans: [], sibling_events: [sibling] } }));
    renderDetail();

    const note = await screen.findByTestId("judged-after-rewrite");
    expect(note).toHaveTextContent("Judged after mask_emails rewrote it.");
  });

  it("explains an observe-only mask and labels its diff as not applied", async () => {
    mockFetch(
      detail(
        event({
          outcome: "warned",
          effect: "observed",
          action: "mask",
          action_taken: "masked",
          blocking: false,
          metadata: { before: "call 555-1234", after: "call [PHONE]" },
        })
      )
    );
    renderDetail();

    const panel = await screen.findByTestId("panel-outcome");
    expect(
      within(panel).getByText("Observe-only — would have masked; nothing applied.")
    ).toBeInTheDocument();
    expect(within(panel).getByText("would have produced (not applied)")).toBeInTheDocument();
    expect(within(panel).getByText("call [PHONE]")).toBeInTheDocument();
    expect(within(screen.getByTestId("panel-rule")).getByText("observe-only")).toBeInTheDocument();
  });

  it("says an errored observer enforced nothing", async () => {
    mockFetch(
      detail(
        event({
          outcome: "errored",
          effect: "observed",
          blocking: false,
          action_taken: "none",
          metadata: { error: "judge timed out", on_error: "block" },
        })
      )
    );
    renderDetail();

    const panel = await screen.findByTestId("panel-outcome");
    expect(
      within(panel).getByText(
        "Could not run — observe-only: nothing was enforced, content passed (on_error applies only to blocking rules)."
      )
    ).toBeInTheDocument();
    expect(screen.getByText("judge timed out")).toBeInTheDocument();
  });

  it("shows the rule's mode, on_error, severity, floor, origin and latency", async () => {
    mockFetch(detail(event({ attempt: 2 })));
    renderDetail();

    const panel = await screen.findByTestId("panel-rule");
    expect(within(panel).getByText("blocking")).toBeInTheDocument();
    expect(within(panel).getByText("on_error:")).toBeInTheDocument();
    expect(within(panel).getByText("block")).toBeInTheDocument();
    expect(within(panel).getByText("high")).toBeInTheDocument();
    expect(within(panel).getByText("floor")).toBeInTheDocument();
    expect(within(panel).getByText("plane")).toBeInTheDocument();
    expect(within(panel).getByText("12 ms")).toBeInTheDocument();
    expect(within(panel).getByText("2")).toBeInTheDocument();
    expect(within(panel).getByTestId("rule-action-chain")).toHaveTextContent(
      "block → blocked → blocked"
    );
  });

  it("renders content_safety evidence per category", async () => {
    mockFetch(
      detail(
        event({
          guardrail_type: "content_safety",
          metadata: {
            taxonomy: "mlcommons",
            scores: { S1: 0.91, S2: 0.1 },
            thresholds: { S1: 0.5 },
            tripped: ["S1"],
            unscored: ["S10"],
          },
        })
      )
    );
    renderDetail();

    const evidence = await screen.findByTestId("evidence-content-safety");
    expect(within(evidence).getByText("0.91")).toBeInTheDocument();
    expect(within(evidence).getByText("0.50")).toBeInTheDocument();
    expect(within(evidence).getByText("default")).toBeInTheDocument();
    expect(within(evidence).getByText("yes")).toBeInTheDocument();
    expect(within(evidence).getByText("S10")).toBeInTheDocument();
  });

  it("renders groundedness evidence with the unsupported claims", async () => {
    mockFetch(
      detail(
        event({
          guardrail_type: "groundedness",
          metadata: {
            score: 0.4,
            threshold: 0.7,
            unsupported_claims: ["The refund window is 90 days"],
          },
        })
      )
    );
    renderDetail();

    const evidence = await screen.findByTestId("evidence-groundedness");
    expect(within(evidence).getByText("0.40")).toBeInTheDocument();
    expect(within(evidence).getByText("0.70")).toBeInTheDocument();
    expect(within(evidence).getByText("The refund window is 90 days")).toBeInTheDocument();
  });

  it("renders schema violations as a table", async () => {
    mockFetch(
      detail(
        event({
          guardrail_type: "schema",
          metadata: {
            violations: [
              { path: "order/total", message: "'x' is not of type 'number'", validator: "type" },
            ],
          },
        })
      )
    );
    renderDetail();

    const evidence = await screen.findByTestId("evidence-schema");
    expect(within(evidence).getByText("order/total")).toBeInTheDocument();
    expect(within(evidence).getByText("'x' is not of type 'number'")).toBeInTheDocument();
    expect(within(evidence).getByText("type")).toBeInTheDocument();
  });

  it("falls back to the raw metadata for a custom code rule", async () => {
    mockFetch(
      detail(event({ guardrail_type: "code", metadata: { custom_signal: "lookalike-domain" } }))
    );
    renderDetail();

    const evidence = await screen.findByTestId("panel-evidence");
    expect(
      within(evidence).getByText(/No type-specific evidence for this rule/)
    ).toBeInTheDocument();
    expect(within(evidence).getByText("All recorded metadata")).toBeInTheDocument();
    expect(within(evidence).getByTestId("evidence-raw-metadata")).toHaveTextContent(
      "lookalike-domain"
    );
    // Keys no runner writes are never promised.
    expect(screen.queryByText(/judge prompt/i)).not.toBeInTheDocument();
  });

  it("sends ?redact=true when Mask secrets is on", async () => {
    const spy = mockFetch(detail(event()));
    renderDetail();

    await screen.findByTestId("panel-trigger");
    fireEvent.click(screen.getByRole("switch", { name: /redaction/i }));
    await waitFor(() =>
      expect(
        spy.mock.calls.some(([url]) => String(url).includes("/guardrail-events/e1?redact=true"))
      ).toBe(true)
    );
  });
});
