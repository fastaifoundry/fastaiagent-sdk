import { describe, expect, it } from "vitest";
import { buildAttention, type AttentionInput } from "./attention";

const CLEAN: AttentionInput = {
  failing_traces_last_24h: 0,
  eval_runs_last_7d: 0,
  avg_pass_rate_last_7d: 0,
  eval_case_pass_rate_7d: null,
  eval_datasets_7d: 0,
  eval_runs_counted_7d: 0,
  eval_gate_failures_7d: 0,
  pending_approvals_count: 0,
  failed_executions_count: 0,
  agents_with_errors: [],
};

const at = (over: Partial<AttentionInput>) => buildAttention({ ...CLEAN, ...over });

describe("buildAttention", () => {
  it("returns nothing when everything is healthy", () => {
    expect(at({})).toEqual([]);
  });

  it("returns nothing when the payload has not loaded", () => {
    expect(buildAttention(undefined)).toEqual([]);
  });

  it("does NOT flag a null cases rate (no case reached a verdict)", () => {
    // Nothing judged is not a regression — flagging it would make an idle
    // project look broken.
    expect(at({ eval_runs_last_7d: 3, eval_case_pass_rate_7d: null })).toEqual([]);
  });

  it("flags a low cases rate", () => {
    const items = at({ eval_runs_last_7d: 4, eval_case_pass_rate_7d: 0.5, eval_datasets_7d: 2 });
    expect(items).toHaveLength(1);
    expect(items[0].key).toBe("low-pass-rate");
    expect(items[0].title).toContain("50%");
    expect(items[0].detail).toContain("2 datasets");
    expect(items[0].to).toBe("/evals");
  });

  it("ignores the legacy checks average for the warning", () => {
    // avg_pass_rate_last_7d counted dataset checks and AutoLLM candidates.
    expect(
      at({ eval_runs_last_7d: 4, avg_pass_rate_last_7d: 0.1, eval_case_pass_rate_7d: 0.95 })
    ).toEqual([]);
  });

  it("does not flag a cases rate at or above the floor", () => {
    expect(at({ eval_runs_last_7d: 4, eval_case_pass_rate_7d: 0.7 })).toEqual([]);
    expect(at({ eval_runs_last_7d: 4, eval_case_pass_rate_7d: 0.95 })).toEqual([]);
  });

  it("flags gated runs that failed or were invalid, linking to the filtered list", () => {
    const items = at({ eval_gate_failures_7d: 2 });
    expect(items).toHaveLength(1);
    expect(items[0].key).toBe("gate-failures");
    expect(items[0].title).toBe("2 gated eval runs failed or were invalid");
    expect(items[0].to).toBe("/evals?gate=failed,invalid");
  });

  it("uses singular wording for one failed gate", () => {
    const [item] = at({ eval_gate_failures_7d: 1 });
    expect(item.title).toBe("1 gated eval run failed or was invalid");
  });

  it("ranks critical above warning", () => {
    const items = at({
      failed_executions_count: 2,
      failing_traces_last_24h: 9,
      pending_approvals_count: 1,
    });
    expect(items.map((i) => i.severity)).toEqual([
      "critical",
      "warning",
      "warning",
    ]);
    expect(items[0].key).toBe("failed-executions");
  });

  it("names the worst-offending agents in the failing-traces detail", () => {
    const items = at({
      failing_traces_last_24h: 6,
      agents_with_errors: [
        { agent_name: "quiet-bot", error_count: 1 },
        { agent_name: "support-bot", error_count: 4 },
        { agent_name: "triage-bot", error_count: 2 },
      ],
    });
    const failing = items.find((i) => i.key === "failing-traces")!;
    // Sorted by error count, worst first.
    expect(failing.detail).toContain("support-bot, triage-bot, quiet-bot");
  });

  it("caps the named agents at three", () => {
    const items = at({
      failing_traces_last_24h: 20,
      agents_with_errors: Array.from({ length: 6 }, (_, i) => ({
        agent_name: `bot-${i}`,
        error_count: 10 - i,
      })),
    });
    const failing = items.find((i) => i.key === "failing-traces")!;
    expect(failing.detail).toContain("bot-0, bot-1, bot-2");
    expect(failing.detail).not.toContain("bot-3");
  });

  it("falls back to generic wording when no agent name was recorded", () => {
    const items = at({ failing_traces_last_24h: 3, agents_with_errors: [] });
    const failing = items.find((i) => i.key === "failing-traces")!;
    expect(failing.detail).toContain("filtered to errors");
  });

  it("uses singular wording for a count of one", () => {
    const [item] = at({ pending_approvals_count: 1 });
    expect(item.title).toBe("1 run is waiting for approval");
  });

  it("uses plural wording for counts above one", () => {
    const [item] = at({ pending_approvals_count: 3 });
    expect(item.title).toBe("3 runs are waiting for approval");
  });

  it("gives every item a route the app can navigate to", () => {
    const items = at({
      failed_executions_count: 1,
      failing_traces_last_24h: 1,
      pending_approvals_count: 1,
      eval_runs_last_7d: 2,
      eval_case_pass_rate_7d: 0.1,
      eval_gate_failures_7d: 1,
    });
    expect(items).toHaveLength(5);
    for (const item of items) {
      expect(item.to).toMatch(/^\//);
      expect(item.cta.length).toBeGreaterThan(0);
    }
    // Keys are unique, so React list rendering is stable.
    expect(new Set(items.map((i) => i.key)).size).toBe(items.length);
  });
});
