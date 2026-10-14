import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { SpanTree } from "./SpanTree";
import type { SpanTreeNode } from "@/lib/types";

function span(
  id: string,
  name: string,
  start: number,
  end: number,
  children: SpanTreeNode[] = []
): SpanTreeNode {
  return {
    span: {
      span_id: id,
      trace_id: "t",
      parent_span_id: null,
      name,
      start_time: new Date(start).toISOString(),
      end_time: new Date(end).toISOString(),
      status: "OK",
      attributes: {},
      events: [],
    },
    children,
  };
}

describe("SpanTree", () => {
  const tree = span("root", "agent.example", 0, 1000, [
    span("child-1", "llm.chat", 100, 500),
    span("child-2", "tool.search", 500, 900, [
      span("grandchild", "retrieval.lookup", 550, 870),
    ]),
  ]);

  it("renders every span with its name", () => {
    render(
      <SpanTree tree={tree} selectedSpanId={null} onSelect={() => {}} />
    );
    expect(screen.getByText("agent.example")).toBeInTheDocument();
    expect(screen.getByText("llm.chat")).toBeInTheDocument();
    expect(screen.getByText("tool.search")).toBeInTheDocument();
    expect(screen.getByText("retrieval.lookup")).toBeInTheDocument();
  });

  it("calls onSelect with the clicked span", () => {
    const onSelect = vi.fn();
    render(
      <SpanTree tree={tree} selectedSpanId={null} onSelect={onSelect} />
    );
    fireEvent.click(screen.getByText("tool.search"));
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ span_id: "child-2", name: "tool.search" })
    );
  });

  it("renders ERR chip for failed spans", () => {
    const failing = span("x", "agent.failing", 0, 100);
    failing.span.status = "ERROR";
    render(<SpanTree tree={failing} selectedSpanId={null} onSelect={() => {}} />);
    expect(screen.getByText("err")).toBeInTheDocument();
  });

  // 1.89.0 — a guardrail span is coloured by what the check did, never by
  // its OTel status (ERROR for any non-pass verdict, a warn included).
  it("shows a fail-open guardrail span with status OK as unchecked, not green", () => {
    const child = span("g", "guardrail.no_pii", 10, 20);
    child.span.status = "OK";
    child.span.attributes = {
      "fastaiagent.guardrail.passed": true,
      "fastaiagent.guardrail.errored": true,
    };
    const root = span("root", "agent.example", 0, 100, [child]);
    render(<SpanTree tree={root} selectedSpanId={null} onSelect={() => {}} />);
    const chip = screen.getByTestId("guardrail-verdict");
    expect(chip).toHaveTextContent(/unchecked/i);
    expect(chip.className).not.toMatch(/fa-success/);
    expect(screen.queryByText("err")).not.toBeInTheDocument();
  });

  it("shows a warn guardrail span with status ERROR as warned, with no err chip", () => {
    const child = span("g", "guardrail.tone", 10, 20);
    child.span.status = "ERROR";
    child.span.attributes = {
      "fastaiagent.guardrail.passed": false,
      "fastaiagent.guardrail.action_taken": "warned",
    };
    const root = span("root", "agent.example", 0, 100, [child]);
    render(<SpanTree tree={root} selectedSpanId={null} onSelect={() => {}} />);
    expect(screen.getByTestId("guardrail-verdict")).toHaveTextContent(/warned/i);
    expect(screen.queryByText("err")).not.toBeInTheDocument();
  });

  it("prefers the event row's effect over the span attributes", () => {
    const child = span("g", "guardrail.no_pii", 10, 20);
    child.span.status = "ERROR";
    child.span.attributes = {
      "fastaiagent.guardrail.passed": false,
      "fastaiagent.guardrail.action_taken": "masked",
    };
    child.span.guardrail = {
      event_id: "e1",
      outcome: "warned",
      effect: "observed",
      blocking: false,
      final_reason: null,
    };
    render(<SpanTree tree={child} selectedSpanId={null} onSelect={() => {}} />);
    expect(screen.getByTestId("guardrail-verdict")).toHaveTextContent(/observed/i);
    expect(screen.queryByText("err")).not.toBeInTheDocument();
  });
});
