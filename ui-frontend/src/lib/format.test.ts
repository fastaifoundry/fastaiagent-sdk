import { describe, expect, it } from "vitest";
import {
  formatCost,
  formatDurationMs,
  formatTimeAgo,
  formatTokens,
  shortTraceId,
} from "./format";

describe("formatDurationMs", () => {
  it("handles null/undefined", () => {
    expect(formatDurationMs(null)).toBe("—");
    expect(formatDurationMs(undefined)).toBe("—");
  });

  it("formats sub-ms durations", () => {
    expect(formatDurationMs(0.4)).toBe("<1ms");
  });

  it("formats ms", () => {
    expect(formatDurationMs(47)).toBe("47ms");
    expect(formatDurationMs(999)).toBe("999ms");
  });

  it("formats seconds with sensible precision", () => {
    expect(formatDurationMs(1234)).toBe("1.23s");
    expect(formatDurationMs(12345)).toBe("12.3s");
  });

  it("formats minutes + seconds", () => {
    expect(formatDurationMs(65_000)).toBe("1m 5s");
    expect(formatDurationMs(180_000)).toBe("3m 0s");
  });
});

describe("formatCost", () => {
  it("treats null and zero as dash", () => {
    expect(formatCost(null)).toBe("—");
    expect(formatCost(0)).toBe("—");
  });

  it("formats sub-millicent as millicents", () => {
    expect(formatCost(0.0005)).toBe("$0.50m");
  });

  it("formats cents with four decimals", () => {
    expect(formatCost(0.01234)).toBe("$0.0123");
  });

  it("formats dollars with two decimals", () => {
    expect(formatCost(1.5)).toBe("$1.50");
    expect(formatCost(123.456)).toBe("$123.46");
  });
});

describe("formatTokens", () => {
  it("passes through small counts", () => {
    expect(formatTokens(0)).toBe("0");
    expect(formatTokens(999)).toBe("999");
  });

  it("abbreviates thousands", () => {
    expect(formatTokens(1234)).toBe("1.2k");
  });

  it("abbreviates millions", () => {
    expect(formatTokens(1_200_000)).toBe("1.2M");
  });

  it("handles null", () => {
    expect(formatTokens(null)).toBe("—");
  });
});

describe("formatTimeAgo", () => {
  it("returns dash for empty", () => {
    expect(formatTimeAgo(null)).toBe("—");
    expect(formatTimeAgo("")).toBe("—");
  });

  it("returns 'just now' for <10s", () => {
    const now = new Date().toISOString();
    expect(formatTimeAgo(now)).toBe("just now");
  });

  it("returns seconds past for >=10s", () => {
    const iso = new Date(Date.now() - 45_000).toISOString();
    expect(formatTimeAgo(iso)).toBe("45s ago");
  });

  it("returns minutes past when appropriate", () => {
    const iso = new Date(Date.now() - 7 * 60_000).toISOString();
    expect(formatTimeAgo(iso)).toBe("7m ago");
  });

  it("returns hours past when appropriate", () => {
    const iso = new Date(Date.now() - 3 * 60 * 60_000).toISOString();
    expect(formatTimeAgo(iso)).toBe("3h ago");
  });
});

describe("shortTraceId", () => {
  it("passes short ids through", () => {
    expect(shortTraceId("abc")).toBe("abc");
  });

  it("collapses long ids", () => {
    expect(shortTraceId("abcdef0123456789fedcba")).toBe("abcdef01…dcba");
  });
});
