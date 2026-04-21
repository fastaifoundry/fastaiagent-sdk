import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EventsPane } from "./EventsPane";
import type { SpanEvent } from "@/lib/types";

describe("EventsPane", () => {
  it("shows a friendly explainer when no events are attached", () => {
    render(<EventsPane events={[]} />);
    expect(
      screen.getByText(/No events emitted/i)
    ).toBeInTheDocument();
    // Explainer should also describe what events ARE, since users may land
    // here without knowing.
    expect(
      screen.getByText(/timestamped occurrences/i)
    ).toBeInTheDocument();
  });

  it("pretty-prints an exception event with type, message, and traceback", async () => {
    const event: SpanEvent = {
      name: "exception",
      timestamp: "1776722007568361000",
      attributes: {
        "exception.type": "ValueError",
        "exception.message": "input must be non-empty",
        "exception.stacktrace":
          'Traceback (most recent call last):\n  File "agent.py", line 42, in run\n    raise ValueError("input must be non-empty")\nValueError: input must be non-empty\n',
        "exception.escaped": true,
      },
    };
    const user = userEvent.setup();
    render(<EventsPane events={[event]} />);

    // Type prominent, message visible, escaped flag surfaced.
    // Message text appears twice (once in the message div, once embedded in
    // the stacktrace) — first() picks the dedicated message row.
    expect(screen.getByText("ValueError")).toBeInTheDocument();
    expect(screen.getAllByText(/input must be non-empty/)[0]).toBeInTheDocument();
    expect(screen.getByText(/escaped/i)).toBeInTheDocument();

    // Traceback is hidden behind a disclosure — expand then verify it's there.
    const summary = screen.getByText(/^Traceback$/i);
    await user.click(summary);
    expect(
      screen.getByText(/raise ValueError/)
    ).toBeInTheDocument();
  });

  it("renders a generic (non-exception) event with name and JSON attributes", () => {
    const event: SpanEvent = {
      name: "handoff",
      timestamp: "1776722007568361000",
      attributes: { from: "triage", to: "coder" },
    };
    render(<EventsPane events={[event]} />);
    expect(screen.getByText("handoff")).toBeInTheDocument();
  });

  it("falls back to the raw timestamp string when it isn't a nanosecond epoch", () => {
    const event: SpanEvent = {
      name: "custom",
      timestamp: "not-a-number",
      attributes: {},
    };
    render(<EventsPane events={[event]} />);
    expect(screen.getByText("not-a-number")).toBeInTheDocument();
  });
});
