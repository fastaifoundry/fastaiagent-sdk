import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { ReplayOriginBanner } from "./ReplayOriginBanner";
import type { SpanRow } from "@/lib/types";

function span(
  id: string,
  parent: string | null,
  attributes: Record<string, unknown>
): SpanRow {
  return {
    span_id: id,
    trace_id: "t2",
    parent_span_id: parent,
    name: "agent.support-bot",
    start_time: new Date().toISOString(),
    end_time: new Date().toISOString(),
    status: "OK",
    attributes,
    events: [],
  };
}

describe("ReplayOriginBanner", () => {
  it("links a rerun's trace to the trace it replayed (1.89.0)", () => {
    renderWithProviders(
      <ReplayOriginBanner
        spans={[
          span("root", null, {
            "fastaiagent.source": "replay",
            "fastaiagent.replay.source_trace_id": "abcdef0123456789",
            "fastaiagent.replay.fork_step": 2,
            "fastaiagent.replay.determinism": "recorded",
            "fastaiagent.replay.tool_policy": "block",
            "fastaiagent.replay.surface": "ui",
          }),
          span("child", "root", {}),
        ]}
      />
    );
    const banner = screen.getByTestId("replay-origin-banner");
    expect(banner).toHaveTextContent("Replay of abcdef01…6789");
    expect(banner).toHaveTextContent("determinism: recorded");
    expect(banner).toHaveTextContent("tool policy: block");
    expect(banner).toHaveTextContent("fork step 2");
    expect(screen.getByRole("link", { name: "abcdef01…6789" })).toHaveAttribute(
      "href",
      "/traces/abcdef0123456789"
    );
  });

  it("renders nothing for a trace that is not a replay", () => {
    renderWithProviders(
      <ReplayOriginBanner spans={[span("root", null, { "fastaiagent.source": "playground" })]} />
    );
    expect(screen.queryByTestId("replay-origin-banner")).not.toBeInTheDocument();
  });
});
