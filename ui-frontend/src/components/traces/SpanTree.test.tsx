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
});
