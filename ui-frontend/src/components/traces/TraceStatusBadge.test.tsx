import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { TraceStatusBadge } from "./TraceStatusBadge";

describe("TraceStatusBadge", () => {
  it("renders OK for clean statuses", () => {
    render(<TraceStatusBadge status="OK" />);
    expect(screen.getByText("OK")).toBeInTheDocument();
  });

  it("upper-cases whatever status comes in", () => {
    render(<TraceStatusBadge status="error" />);
    expect(screen.getByText("ERROR")).toBeInTheDocument();
  });

  it("normalizes UNSET → OK", () => {
    render(<TraceStatusBadge status="UNSET" />);
    expect(screen.getByText("OK")).toBeInTheDocument();
  });

  it("falls through to warning styling on unknown statuses", () => {
    const { container } = render(<TraceStatusBadge status="WEIRD" />);
    expect(screen.getByText("WEIRD")).toBeInTheDocument();
    // Dot picked the warning class.
    expect(container.querySelector(".bg-fa-warning")).not.toBeNull();
  });
});
