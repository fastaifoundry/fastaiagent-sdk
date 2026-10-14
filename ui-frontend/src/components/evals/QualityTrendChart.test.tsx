import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { QualityTrendChart } from "./QualityTrendChart";

describe("QualityTrendChart", () => {
  it("says there is no trend when no run reached a case verdict (no 0% line)", () => {
    render(
      <QualityTrendChart
        points={[
          { started_at: "2026-10-01T00:00:00Z", dataset_name: "a", series: "a", case_pass_rate: null, pass_rate: null },
        ]}
      />
    );
    expect(screen.getByText("Not enough runs to draw a trend yet.")).toBeInTheDocument();
  });

  it("mounts one series per dataset with a legend", () => {
    render(
      <QualityTrendChart
        points={[
          { started_at: "2026-10-01T00:00:00Z", dataset_name: "qa", series: "qa", case_pass_rate: 0.5, pass_rate: 0.6 },
          { started_at: "2026-10-02T00:00:00Z", dataset_name: "billing", series: "billing", case_pass_rate: 1, pass_rate: 1 },
        ]}
      />
    );
    expect(screen.getByText("qa")).toBeInTheDocument();
    expect(screen.getByText("billing")).toBeInTheDocument();
    expect(screen.getByText(/Cases pass rate/)).toBeInTheDocument();
  });
});
