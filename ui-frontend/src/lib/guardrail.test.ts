import { describe, expect, it } from "vitest";
import {
  actionSummary,
  effectSuffix,
  guardrailNarrative,
  guardrailSpanVerdict,
  outcomeMeta,
} from "./guardrail";
import type { GuardrailEvent, SpanRow } from "./types";

function span(
  attributes: Record<string, unknown>,
  extra: Partial<SpanRow> = {}
): SpanRow {
  return {
    span_id: "s",
    trace_id: "t",
    parent_span_id: "p",
    name: "guardrail.no_pii",
    start_time: new Date(0).toISOString(),
    end_time: new Date(10).toISOString(),
    status: "OK",
    attributes,
    events: [],
    ...extra,
  };
}

function event(overrides: Partial<GuardrailEvent> = {}): GuardrailEvent {
  return {
    event_id: "e1",
    trace_id: "t1",
    span_id: "s1",
    guardrail_name: "no_pii",
    guardrail_type: "pii",
    position: "output",
    outcome: "passed",
    score: null,
    message: null,
    agent_name: "support",
    timestamp: new Date().toISOString(),
    metadata: {},
    false_positive: false,
    false_positive_at: null,
    action: "block",
    action_taken: "none",
    severity: null,
    floor: false,
    effect: "none",
    final_reason: null,
    latency_ms: 3,
    attempt: null,
    batch_id: "b1",
    guardrail_span_id: "gs1",
    judged_ref: "ref",
    rule: null,
    blocking: true,
    on_error: "block",
    origin: "local",
    ...overrides,
  };
}

const text = (ev: GuardrailEvent) => guardrailNarrative(ev).lines.join(" ");

describe("guardrailSpanVerdict", () => {
  it("is null for spans that are not guardrails", () => {
    expect(guardrailSpanVerdict(span({}, { name: "llm.chat" }))).toBeNull();
  });

  it.each([
    ["none", "pass", "success"],
    ["warned", "warned", "warning"],
    ["rewritten", "rewrote", "info"],
    ["reasked", "re-asked", "warning"],
    ["blocked", "blocked", "danger"],
    ["observed", "observed", "neutral"],
    ["unchecked", "unchecked", "warning"],
  ] as const)("maps effect %s to %s", (effect, label, tone) => {
    const s = span(
      { "fastaiagent.guardrail.passed": true },
      {
        // The enrichment wins over status and attributes alike.
        status: "ERROR",
        guardrail: {
          event_id: "e1",
          outcome: null,
          effect,
          blocking: true,
          final_reason: null,
        },
      }
    );
    expect(guardrailSpanVerdict(s)).toEqual({ label, tone });
  });

  it.each([
    [{ passed: true, errored: true }, "unchecked", "warning"],
    [{ passed: false, errored: true }, "could not run", "warning"],
    [{ passed: true }, "pass", "success"],
    [{ passed: false, action_taken: "warned" }, "warned", "warning"],
    [{ passed: false, action_taken: "masked" }, "rewrote", "info"],
    [{ passed: false, action_taken: "overridden" }, "rewrote", "info"],
    [{ passed: false, action_taken: "reask" }, "re-asked", "warning"],
    [{ passed: false, action_taken: "blocked" }, "blocked", "danger"],
  ])("derives %o from span attributes", (attrs, label, tone) => {
    const prefixed = Object.fromEntries(
      Object.entries(attrs).map(([k, v]) => [`fastaiagent.guardrail.${k}`, v])
    );
    expect(guardrailSpanVerdict(span(prefixed))).toEqual({ label, tone });
  });

  it("reads string-typed boolean attributes", () => {
    const s = span({
      "fastaiagent.guardrail.passed": "true",
      "fastaiagent.guardrail.errored": "true",
    });
    expect(guardrailSpanVerdict(s)?.label).toBe("unchecked");
  });

  it("never reads OTel status: a fail-open with status OK is not a pass", () => {
    const s = span(
      { "fastaiagent.guardrail.passed": true, "fastaiagent.guardrail.errored": true },
      { status: "OK" }
    );
    expect(guardrailSpanVerdict(s)?.tone).not.toBe("success");
  });
});

describe("OUTCOME_META / effectSuffix", () => {
  it("labels errored as could-not-run in amber, and maps reasked and filtered", () => {
    expect(outcomeMeta("errored")).toEqual({ label: "could not run", tone: "warning" });
    expect(outcomeMeta("reasked")).toEqual({ label: "re-asked", tone: "warning" });
    expect(outcomeMeta("filtered").tone).toBe("info");
    expect(outcomeMeta("weird")).toEqual({ label: "weird", tone: "neutral" });
  });

  it("adds an effect only when the outcome does not already imply it", () => {
    expect(effectSuffix("blocked", "blocked")).toBeNull();
    expect(effectSuffix("warned", "observed")).toBe("observe-only");
    expect(effectSuffix("errored", "unchecked")).toBe("unchecked");
    expect(effectSuffix("passed", null)).toBeNull();
  });
});

describe("actionSummary", () => {
  it("says an observe-only mask applied nothing", () => {
    const s = actionSummary(
      event({ outcome: "warned", action: "mask", action_taken: "masked", effect: "observed" })
    );
    expect(s?.text).toBe("masked · observe-only");
    expect(s?.diverged).toBe(false);
  });

  it("shows a refused rewrite as a block and carries final_reason", () => {
    const s = actionSummary(
      event({
        outcome: "blocked",
        action: "mask",
        action_taken: "masked",
        effect: "blocked",
        final_reason: "masked stream already yielded",
      })
    );
    expect(s).toEqual({
      text: "blocked (was masked)",
      diverged: true,
      title: "masked stream already yielded",
    });
  });

  it("is null when nothing was done", () => {
    expect(actionSummary(event())).toBeNull();
  });
});

describe("guardrailNarrative", () => {
  it.each([
    [
      { outcome: "errored", effect: "unchecked" as const },
      "Could not run — allowed through unchecked (on_error=allow). A degraded pass, not a verdict.",
    ],
    [
      { outcome: "errored", effect: "blocked" as const },
      "Could not run — blocked (fail closed).",
    ],
    [
      { outcome: "errored", effect: "observed" as const, blocking: false },
      "Could not run — observe-only: nothing was enforced, content passed (on_error applies only to blocking rules).",
    ],
  ])("errored %o", (overrides, expected) => {
    expect(text(event(overrides))).toBe(expected);
  });

  it("blocked: the run stopped, with the configured action, final reason and framework", () => {
    const n = text(
      event({
        outcome: "blocked",
        effect: "blocked",
        action: "mask",
        action_taken: "masked",
        final_reason: "rewrite refused by caller",
        metadata: { framework: "langchain", side: "output" },
      })
    );
    expect(n).toContain("Run stopped here.");
    expect(n).toContain("Configured to mask; rewrite refused by caller");
    expect(n).toContain("langchain framework call raised GuardrailBlocked");
  });

  it("reasked names the attempt", () => {
    expect(text(event({ outcome: "reasked", effect: "reasked", attempt: 2 }))).toBe(
      "Model re-asked (attempt 2); see the next attempt's events."
    );
  });

  it("observed says what it would have done and that nothing applied", () => {
    expect(
      text(event({ outcome: "warned", effect: "observed", action_taken: "masked", blocking: false }))
    ).toBe("Observe-only — would have masked; nothing applied.");
  });

  it("a blocking warn keeps the warned text", () => {
    expect(text(event({ outcome: "warned", effect: "warned", action_taken: "warned" }))).toMatch(
      /warned but did not block/
    );
  });

  it("a clean pass", () => {
    expect(text(event())).toMatch(/passed — content was allowed/);
  });

  describe("legacy rows (effect null)", () => {
    it("errored without on_error=allow says blocked only if the rule was blocking", () => {
      const n = text(
        event({ outcome: "errored", effect: null, on_error: null, blocking: null, metadata: {} })
      );
      expect(n).toContain("blocked (fail closed) if this rule was blocking");
    });

    it("errored with on_error=allow in metadata is a fail open", () => {
      const n = text(
        event({
          outcome: "errored",
          effect: null,
          on_error: null,
          metadata: { on_error: "allow" },
        })
      );
      expect(n).toContain("allowed through (fail open)");
    });

    it.each([
      ["blocked", /blocked this content/],
      ["filtered", /rewrote the content/],
      ["warned", /warned but did not block/],
      ["passed", /passed — content was allowed/],
    ])("%s keeps today's narrative", (outcome, expected) => {
      expect(text(event({ outcome, effect: null }))).toMatch(expected);
    });
  });
});
