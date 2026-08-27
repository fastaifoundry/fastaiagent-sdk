import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Timestamp } from "./Timestamp";
import { formatDateTime } from "@/lib/format";

const ISO = "2026-08-27T21:44:53+00:00";

describe("Timestamp", () => {
  it("renders the absolute time, not a relative one", () => {
    render(<Timestamp iso={ISO} />);
    expect(screen.getByText(formatDateTime(ISO))).toBeInTheDocument();
    expect(screen.queryByText(/ago/)).toBeNull();
  });

  it("keeps the relative reading available in the tooltip", () => {
    render(<Timestamp iso={ISO} />);
    const title = screen.getByText(formatDateTime(ISO)).getAttribute("title")!;
    // Full date (with year) plus the relative form — nothing is lost by
    // switching the visible value to absolute.
    expect(title).toContain("2026");
    expect(title).toMatch(/ago|just now/);
  });

  it("is tabular so a column of timestamps stays aligned", () => {
    render(<Timestamp iso={ISO} />);
    const el = screen.getByText(formatDateTime(ISO));
    expect(el.className).toContain("tabular-nums");
    expect(el.className).toContain("font-mono");
  });

  it("renders a dash and no tooltip when there is no timestamp", () => {
    render(<Timestamp iso={null} />);
    const el = screen.getByText("—");
    expect(el).toBeInTheDocument();
    expect(el.getAttribute("title")).toBeNull();
  });

  it("distinguishes two events one second apart", () => {
    const { container: a } = render(
      <Timestamp iso="2026-08-27T21:44:53+00:00" />
    );
    const { container: b } = render(
      <Timestamp iso="2026-08-27T21:44:54+00:00" />
    );
    expect(a.textContent).not.toBe(b.textContent);
  });
});
