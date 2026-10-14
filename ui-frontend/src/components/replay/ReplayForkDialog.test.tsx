import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/test/utils";
import { ReplayForkDialog } from "./ReplayForkDialog";
import type { ReplayStep, ReplayToolChoice } from "@/lib/types";

/**
 * The fork dialog's per-tool choices become the rerun's tool plan (1.89.0):
 * Live → ``allow_tools``, Stub → ``tool_overrides``, everything else stays
 * blocked under ``tool_policy: "block"``. The api.ts boundary (fetch) is the
 * only mock surface.
 */

const STEP: ReplayStep = {
  step: 0,
  span_name: "agent.support-bot",
  span_id: "span-0",
  input: {},
  output: {},
  attributes: {},
  timestamp: new Date().toISOString(),
};

function tool(name: string, extra: Partial<ReplayToolChoice> = {}): ReplayToolChoice {
  return {
    name,
    tool_type: "function",
    origin: null,
    replay_class: null,
    live_available: true,
    endpoint_from_trace: false,
    credentials_redacted: false,
    ...extra,
  };
}

const TOOLS: ReplayToolChoice[] = [
  tool("lookup_order"),
  tool("issue_refund"),
  tool("send_email"),
  tool("crm_api", {
    tool_type: "rest_api",
    live_available: false,
    endpoint_from_trace: true,
    credentials_redacted: true,
  }),
  tool("unregistered", { live_available: false }),
];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function stubApi() {
  const patches: Record<string, unknown>[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    const method = (init?.method ?? "GET").toUpperCase();
    if (method === "POST" && url.endsWith("/api/replay/t1/fork")) {
      return json({ fork_id: "f1" });
    }
    if (method === "PATCH" && url.endsWith("/api/replay/forks/f1")) {
      patches.push(JSON.parse(String(init?.body)));
      return json({ fork_id: "f1", status: "modified" });
    }
    if (method === "POST" && url.endsWith("/api/replay/forks/f1/rerun")) {
      return json({
        fork_id: "f1",
        new_trace_id: "t2",
        new_output: "ok",
        original_output: "ok",
        steps_executed: 3,
      });
    }
    return json({});
  });
  return patches;
}

function renderDialog(onRerunComplete = vi.fn()) {
  renderWithProviders(
    <ReplayForkDialog
      traceId="t1"
      step={STEP}
      open
      onOpenChange={() => {}}
      onRerunComplete={onRerunComplete}
      tools={TOOLS}
    />
  );
  return onRerunComplete;
}

afterEach(() => vi.restoreAllMocks());

describe("ReplayForkDialog tool plan", () => {
  it("maps Live to allow_tools and Stub to tool_overrides; the rest stay blocked", async () => {
    const user = userEvent.setup();
    const patches = stubApi();
    const done = renderDialog();

    await user.click(screen.getByRole("tab", { name: "Tools" }));
    await user.click(screen.getByRole("button", { name: "Live lookup_order" }));
    await user.click(screen.getByRole("button", { name: "Stub issue_refund" }));
    fireEvent.change(screen.getByLabelText("Stub response for issue_refund"), {
      target: { value: '{"refund_id": "r-1", "ok": true}' },
    });
    // send_email, crm_api and unregistered are left on the default: Block.
    expect(screen.getByTestId("fork-tool-summary")).toHaveTextContent(
      "Tools: 3 blocked · 1 stubbed · 1 live"
    );

    await user.click(screen.getByRole("button", { name: /Rerun with modifications/ }));
    await waitFor(() => expect(done).toHaveBeenCalled());

    expect(patches).toHaveLength(1);
    expect(patches[0]).toEqual({
      tool_overrides: { issue_refund: { refund_id: "r-1", ok: true } },
      determinism: "live",
      tool_policy: "block",
      allow_tools: ["lookup_order"],
    });
  });

  it("sends a stub that is not JSON as plain text", async () => {
    const user = userEvent.setup();
    const patches = stubApi();
    const done = renderDialog();

    await user.click(screen.getByRole("tab", { name: "Tools" }));
    await user.click(screen.getByRole("button", { name: "Stub send_email" }));
    fireEvent.change(screen.getByLabelText("Stub response for send_email"), {
      target: { value: "queued" },
    });
    await user.click(screen.getByRole("button", { name: /Rerun with modifications/ }));
    await waitFor(() => expect(done).toHaveBeenCalled());

    expect(patches[0].tool_overrides).toEqual({ send_email: "queued" });
    expect(patches[0].allow_tools).toEqual([]);
  });

  it("disables Live for a tool with redacted credentials or no implementation", async () => {
    const user = userEvent.setup();
    stubApi();
    renderDialog();
    await user.click(screen.getByRole("tab", { name: "Tools" }));

    expect(screen.getByRole("button", { name: "Live crm_api" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Live unregistered" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Live lookup_order" })).toBeEnabled();
    expect(
      screen.getByText(/credentials were redacted from the trace, so a live call cannot authenticate/)
    ).toBeInTheDocument();
    expect(screen.getByText(/has no implementation of this tool/)).toBeInTheDocument();
  });

  it("sends on_miss only in recorded mode", async () => {
    const user = userEvent.setup();
    const patches = stubApi();
    const done = renderDialog();

    const onMiss = screen.getByLabelText("When a recorded response is missing");
    expect(onMiss).toBeDisabled();
    fireEvent.change(screen.getByLabelText("LLM calls"), {
      target: { value: "recorded" },
    });
    expect(onMiss).toBeEnabled();
    fireEvent.change(onMiss, { target: { value: "error" } });

    await user.click(screen.getByRole("button", { name: /Rerun with modifications/ }));
    await waitFor(() => expect(done).toHaveBeenCalled());

    expect(patches[0]).toMatchObject({
      determinism: "recorded",
      on_miss: "error",
      tool_policy: "block",
      allow_tools: [],
    });
    expect(patches[0]).not.toHaveProperty("tool_overrides");
  });
});
