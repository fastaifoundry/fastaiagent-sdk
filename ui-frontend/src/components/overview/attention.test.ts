import { describe, expect, it } from "vitest";
import { buildAttention, type AttentionInput } from "./attention";

const CLEAN: AttentionInput = {
  failing_traces_last_24h: 0,
  eval_runs_last_7d: 0,
  avg_pass_rate_last_7d: 0,
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

  it("does NOT flag a 0% pass rate when no evals ran", () => {
    // A 0% average over zero runs is an empty dataset, not a regression —
    // flagging it would make an idle project look broken.
    expect(at({ eval_runs_last_7d: 0, avg_pass_rate_last_7d: 0 })).toEqual([]);
  });

  it("flags a low pass rate once runs exist", () => {
    const items = at({ eval_runs_last_7d: 4, avg_pass_rate_last_7d: 0.5 });
    expect(items).toHaveLength(1);
    expect(items[0].key).toBe("low-pass-rate");
    expect(items[0].title).toContain("50%");
    expect(items[0].to).toBe("/evals");
  });

  it("does not flag a pass rate at or above the floor", () => {
    expect(at({ eval_runs_last_7d: 4, avg_pass_rate_last_7d: 0.7 })).toEqual([]);
    expect(at({ eval_runs_last_7d: 4, avg_pass_rate_last_7d: 0.95 })).toEqual([]);
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
      avg_pass_rate_last_7d: 0.1,
    });
    expect(items).toHaveLength(4);
    for (const item of items) {
      expect(item.to).toMatch(/^\//);
      expect(item.cta.length).toBeGreaterThan(0);
    }
    // Keys are unique, so React list rendering is stable.
    expect(new Set(items.map((i) => i.key)).size).toBe(items.length);
  });
});
