import { describe, expect, it } from "vitest";
import {
  buildTrend,
  caseValueText,
  comparableText,
  costLabel,
  formatCostDelta,
  formatRate,
  formatRateDelta,
  isEchoRun,
  verdictScoreText,
} from "./evalDisplay";

describe("formatRate", () => {
  it("renders — for null, never 0%", () => {
    expect(formatRate(null)).toBe("—");
    expect(formatRate(undefined)).toBe("—");
    expect(formatRate(0)).toBe("0%");
    expect(formatRate(0.8333)).toBe("83%");
  });

  it("signs deltas and keeps unknown as —", () => {
    expect(formatRateDelta(0.05)).toBe("+5%");
    expect(formatRateDelta(-0.1)).toBe("-10%");
    expect(formatRateDelta(null)).toBe("—");
    expect(formatCostDelta(null)).toBe("—");
    expect(formatCostDelta(-0.002)).toBe("-$0.0020");
  });
});

describe("costLabel", () => {
  it("says not traced for a null cost", () => {
    expect(costLabel(null, false)).toEqual({
      text: "—",
      title: "not traced (or no priced model call)",
      partial: false,
    });
  });

  it("flags a partial sum", () => {
    const c = costLabel(0.01, false);
    expect(c.text).toBe("$0.0100");
    expect(c.partial).toBe(true);
  });

  it("shows a known zero as $0", () => {
    expect(costLabel(0, true).text).toBe("$0");
  });
});

describe("caseValueText", () => {
  it("replaces media descriptors with one line, never their JSON", () => {
    const img = {
      type: "image",
      media_type: "image/png",
      size_bytes: 12_500,
      sha256: "abc123def456",
      media_idx: 0,
    };
    expect(caseValueText(img)).toBe("[image · 12 KB · sha256 abc123…]");
    const text = caseValueText(["what is this?", img, { type: "bytes", size_bytes: 10 }]);
    expect(text).toContain("image · 12 KB");
    expect(text).toContain("bytes · 10 B");
    expect(text).not.toContain("media_idx");
  });
});

describe("comparableText", () => {
  it("compares images by sha256 and size, not media_idx or source", () => {
    const a = { type: "image", sha256: "h1", size_bytes: 5, media_idx: 0, source_url: "x" };
    const b = { type: "image", sha256: "h1", size_bytes: 5, media_idx: 3, source_url: null };
    const c = { type: "image", sha256: "h2", size_bytes: 5, media_idx: 0 };
    expect(comparableText(["see", a])).toBe(comparableText(["see", b]));
    expect(comparableText(a)).not.toBe(comparableText(c));
  });
});

describe("verdictScoreText", () => {
  it("hides the placeholder score of errored and n/a verdicts", () => {
    expect(verdictScoreText({ passed: false, score: 0, errored: true })).toBeNull();
    expect(verdictScoreText({ passed: false, score: 0, skipped: true })).toBeNull();
    expect(verdictScoreText({ passed: true, score: 0.9 })).toBe("0.90");
  });
});

describe("buildTrend", () => {
  it("skips points with no cases rate instead of plotting 0", () => {
    const { rows, series } = buildTrend([
      { started_at: "2026-10-01T00:00:00Z", dataset_name: "a", series: "a", case_pass_rate: 0.5, pass_rate: 0.6 },
      { started_at: "2026-10-02T00:00:00Z", dataset_name: "a", series: "a", case_pass_rate: null, pass_rate: null },
      { started_at: "2026-10-03T00:00:00Z", dataset_name: "a", series: "a", case_pass_rate: 1, pass_rate: 1 },
    ]);
    expect(series).toEqual(["a"]);
    expect(rows.map((r) => r.rate)).toEqual([50, 100]);
  });

  it("leaves an invalid (outage) run out of the line instead of a dip", () => {
    const { rows } = buildTrend([
      {
        started_at: "2026-10-01T00:00:00Z",
        dataset_name: "a",
        series: "a",
        case_pass_rate: 0.75,
        pass_rate: 0.8,
      },
      {
        started_at: "2026-10-02T00:00:00Z",
        dataset_name: "a",
        series: "a",
        case_pass_rate: 0,
        pass_rate: 0.75,
        gate_outcome: "invalid",
      },
    ]);
    expect(rows.map((r) => r.rate)).toEqual([75]);
  });

  it("keeps the five series with the most points", () => {
    const points = [];
    for (let s = 0; s < 7; s++) {
      for (let i = 0; i <= s; i++) {
        points.push({
          started_at: `2026-10-0${i + 1}T00:00:00Z`,
          dataset_name: `d${s}`,
          series: `d${s}`,
          case_pass_rate: 0.5,
          pass_rate: 0.5,
        });
      }
    }
    const { series, rows } = buildTrend(points);
    expect(series).toEqual(["d6", "d5", "d4", "d3", "d2"]);
    expect(rows.every((r) => series.includes(r.series))).toBe(true);
  });
});

describe("isEchoRun", () => {
  it("recognises a dataset check by kind or by the echo agent", () => {
    expect(isEchoRun({ kind: "ui-echo", agent_name: "x" })).toBe(true);
    expect(isEchoRun({ kind: null, agent_name: "editor-echo" })).toBe(true);
    expect(isEchoRun({ kind: "ui", agent_name: "support" })).toBe(false);
  });
});
