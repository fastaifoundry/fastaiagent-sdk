import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { PassRateBar } from "./PassRateBar";

describe("PassRateBar", () => {
  it("renders integer percentage", () => {
    render(<PassRateBar passRate={0.75} />);
    expect(screen.getByText("75%")).toBeInTheDocument();
  });

  // A null rate means no verdict applied: "—", never a 0% that reads as all-fail.
  it("renders — for a null rate, with no coloured fill", () => {
    const { container } = render(<PassRateBar passRate={null} />);
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.queryByText("0%")).toBeNull();
    expect(container.querySelector(".bg-destructive")).toBeNull();
  });

  it("still renders a real 0% as 0%", () => {
    render(<PassRateBar passRate={0} passCount={0} failCount={3} />);
    expect(screen.getByText("0%")).toBeInTheDocument();
    expect(screen.getByText("(0/3)")).toBeInTheDocument();
  });

  it("carries a hover title when given one", () => {
    const { container } = render(<PassRateBar passRate={0.5} title="Checks 80%" />);
    expect(container.firstElementChild).toHaveAttribute("title", "Checks 80%");
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
