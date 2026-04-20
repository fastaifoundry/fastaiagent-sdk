import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { PassRateBar } from "./PassRateBar";

describe("PassRateBar", () => {
  it("renders integer percentage", () => {
    render(<PassRateBar passRate={0.75} />);
    expect(screen.getByText("75%")).toBeInTheDocument();
  });

  it("renders 0% for null without exploding", () => {
    render(<PassRateBar passRate={null} />);
    expect(screen.getByText("0%")).toBeInTheDocument();
  });

  it("colors healthy rates green", () => {
    const { container } = render(<PassRateBar passRate={0.95} />);
    expect(container.querySelector(".bg-fa-success")).not.toBeNull();
  });

  it("colors low rates red", () => {
    const { container } = render(<PassRateBar passRate={0.3} />);
    expect(container.querySelector(".bg-destructive")).not.toBeNull();
  });

  it("shows pass/total fraction when counts provided", () => {
    render(<PassRateBar passRate={0.6} passCount={6} failCount={4} />);
    expect(screen.getByText("(6/10)")).toBeInTheDocument();
  });
});
