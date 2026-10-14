/**
 * The Playground page against a stubbed server (1.89.0, P6/P7/P8/P10c–f/P10i).
 *
 * The page streams through `fetch` (POST /api/playground/stream) and reads
 * everything else through api.ts, so `fetch` is the only mock surface: the
 * SSE body below is the exact wire format the server writes.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { Route, Routes } from "react-router-dom";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { PlaygroundPage } from "./PlaygroundPage";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function sse(events: [string, unknown][]): Response {
  const text = events.map(([e, d]) => `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`).join("");
  return new Response(text, { status: 200, headers: { "content-type": "text/event-stream" } });
}

const OPENAI = {
  provider: "openai",
  models: ["gpt-4o-mini"],
  has_key: true,
  env_var: "OPENAI_API_KEY",
  requires_endpoint: false,
};

function done(extra: Record<string, unknown> = {}) {
  return {
    metadata: {
      model: "gpt-4o-mini",
      provider: "openai",
      latency_ms: 12,
      tokens: { input: 3, output: 4 },
      cost_usd: 0.0001,
      trace_id: "t".repeat(32),
      trace_in_ui_db: true,
      finish_reason: "stop",
      truncated: false,
      max_tokens: 1024,
      unfilled_variables: [],
      prompt: null,
      ...extra,
    },
  };
}

interface Stub {
  providers?: unknown[];
  stream?: (body: Record<string, unknown>) => Response;
}

/** Records every request body by path. */
function stubServer({ providers = [OPENAI], stream }: Stub = {}) {
  const bodies: Record<string, Record<string, unknown>[]> = {};
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    const path = url.split("?")[0];
    if (init?.body) (bodies[path] ??= []).push(JSON.parse(String(init.body)));
    if (path === "/api/playground/models") return json({ providers, sampling_rules: [] });
    if (path === "/api/prompts") {
      return json({ rows: [{ name: "greet", latest_version: 3 }], registry_is_local: true });
    }
    if (path === "/api/prompts/greet") {
      return json({
        slug: "greet",
        version: 3,
        template: "Hi {{name}} (v3)",
        template_raw: "Hi {{name}} (v3)",
        fragments: [],
        variables: ["name"],
        metadata: {},
        latest_version: 3,
        registry_is_local: true,
        aliases: { production: 2 },
      });
    }
    if (path === "/api/prompts/greet/versions") {
      return json({
        versions: [1, 2, 3].map((v) => ({ version: String(v), aliases: v === 2 ? ["production"] : [] })),
      });
    }
    const m = path.match(/^\/api\/prompts\/greet\/versions\/(\d+)$/);
    if (m) {
      const v = Number(m[1]);
      return json({ slug: "greet", version: v, template: `Hi {{name}} (v${v})`, variables: ["name"] });
    }
    if (path === "/api/playground/stream") {
      const body = bodies[path]!.at(-1)!;
      return (stream ?? (() => sse([["token", { text: "hello" }], ["done", done()]])))(body);
    }
    if (path === "/api/datasets") return json([]);
    if (path === "/api/playground/save-as-eval") {
      return json({ dataset_name: "playground", path: "/x", line_count: 1 });
    }
    return json({});
  });
  return bodies;
}

function renderPage(route = "/playground") {
  renderWithProviders(
    <Routes>
      <Route path="/playground" element={<PlaygroundPage />} />
    </Routes>,
    { route }
  );
}

const template = () => screen.getByLabelText("Template") as HTMLTextAreaElement;
const runButton = () => screen.getByRole("button", { name: /^Run$/ });

async function typeTemplate(text: string) {
  await waitFor(() => expect(screen.getByText("openai")).toBeInTheDocument());
  fireEvent.change(template(), { target: { value: text } });
}

/** Waits until `n` runs have finished: one history entry each. */
async function finishedRuns(n: number) {
  await waitFor(() => expect(screen.getAllByTitle(/Restore this run/)).toHaveLength(n));
}

afterEach(() => vi.restoreAllMocks());

describe("PlaygroundPage", () => {
  it("warns about unfilled variables before running, and runs anyway on request (P10e)", async () => {
    const bodies = stubServer();
    renderPage();
    await typeTemplate("Hi {{name}}");
    fireEvent.click(runButton());
    expect(await screen.findByTestId("unfilled-warning")).toHaveTextContent("{{name}}");
    expect(bodies["/api/playground/stream"]).toBeUndefined();

    fireEvent.click(screen.getByRole("button", { name: "Run anyway" }));
    await waitFor(() => expect(bodies["/api/playground/stream"]).toHaveLength(1));
    await finishedRuns(1);
  });

  it("shows the server's unfilled list, the truncation banner and the history chip (P10e/f)", async () => {
    stubServer({
      stream: () =>
        sse([
          ["token", { text: "cut" }],
          ["done", done({ truncated: true, finish_reason: "length", max_tokens: 5, unfilled_variables: ["topic"] })],
        ]),
    });
    renderPage();
    await typeTemplate("plain text");
    fireEvent.click(runButton());
    expect(await screen.findByTestId("truncated-banner")).toHaveTextContent(
      "Cut off at max_tokens (5)"
    );
    expect(screen.getByTestId("sent-unfilled")).toHaveTextContent("{{topic}}");
    expect(screen.getByTestId("history-truncated")).toBeInTheDocument();
  });

  it("saves the run's snapshot, not the edited form (P6)", async () => {
    const bodies = stubServer();
    renderPage();
    await typeTemplate("What is 2+2?");
    fireEvent.click(runButton());
    await finishedRuns(1);

    // Edit the form after the run: the case must still pair the old input.
    fireEvent.change(template(), { target: { value: "Something else" } });
    fireEvent.click(screen.getByRole("button", { name: /save as eval case/i }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByTestId("save-eval-form-changed")).toBeInTheDocument();
    expect(within(dialog).getByLabelText("Input (as sent)")).toHaveValue("What is 2+2?");

    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }));
    await waitFor(() => expect(bodies["/api/playground/save-as-eval"]).toHaveLength(1));
    const saved = bodies["/api/playground/save-as-eval"]![0]!;
    expect(saved).toMatchObject({
      input: "What is 2+2?",
      expected_output: "hello",
      model: "gpt-4o-mini",
      provider: "openai",
      trace_id: "t".repeat(32),
      parameters: { temperature: null, top_p: null, max_tokens: 1024 },
    });
    expect(saved).not.toHaveProperty("api_key");
  });

  it("restores the form from a history entry (P10d)", async () => {
    stubServer();
    renderPage();
    await typeTemplate("first prompt");
    fireEvent.click(runButton());
    await finishedRuns(1);
    fireEvent.change(template(), { target: { value: "second prompt" } });
    fireEvent.click(runButton());
    await finishedRuns(2);

    // The newest is first; the older one restores "first prompt".
    fireEvent.click(screen.getAllByTitle(/Restore this run/)[1]!);
    expect(template().value).toBe("first prompt");
  });

  it("loads ?alias= from the registry and sends prompt_ref (P10a/c)", async () => {
    const bodies = stubServer({
      stream: () =>
        sse([
          ["token", { text: "ok" }],
          ["done", done({ prompt: { slug: "greet", version: 2, attributed: true, modified: false } })],
        ]),
    });
    renderPage("/playground?prompt=greet&alias=production");
    // The alias points at v2, so the v2 template is seeded.
    await waitFor(() => expect(template().value).toBe("Hi {{name}} (v2)"));
    fireEvent.change(screen.getByLabelText("{{name}}"), { target: { value: "Ada" } });
    fireEvent.click(runButton());
    await waitFor(() => expect(bodies["/api/playground/stream"]).toHaveLength(1));
    expect(bodies["/api/playground/stream"]![0]!.prompt_ref).toEqual({
      slug: "greet",
      alias: "production",
    });
    expect(await screen.findByTestId("prompt-attribution")).toHaveTextContent(
      "attributed to greet v2"
    );
  });

  it("says when an edited registry template is not attributed (P10c)", async () => {
    stubServer({
      stream: () =>
        sse([
          ["token", { text: "ok" }],
          ["done", done({ prompt: { slug: "greet", version: 3, attributed: false, modified: true } })],
        ]),
    });
    renderPage("/playground?prompt=greet&version=3");
    await waitFor(() => expect(template().value).toBe("Hi {{name}} (v3)"));
    fireEvent.change(template(), { target: { value: "Hey there" } });
    fireEvent.click(runButton());
    expect(await screen.findByTestId("prompt-attribution")).toHaveTextContent(
      "modified from greet v3"
    );
  });

  it("shows the provider's reason for a failed stream (P8)", async () => {
    stubServer({
      stream: () =>
        sse([
          [
            "error",
            {
              message: "LLM call failed.",
              correlation_id: "c0ffee",
              kind: "auth",
              provider_status: 401,
              provider_message: "Incorrect API key provided: [REDACTED].",
              hint: "Check the key for this provider.",
            },
          ],
        ]),
    });
    renderPage();
    await typeTemplate("hi");
    fireEvent.click(runButton());
    expect(await screen.findByTestId("run-error-kind")).toHaveTextContent(
      "Authentication failed (HTTP 401)"
    );
    expect(screen.getByText(/Incorrect API key provided: \[REDACTED\]/)).toBeInTheDocument();
    expect(screen.getByText("c0ffee")).toBeInTheDocument();
  });

  it("holds Run until a provider without a default endpoint gets one (P10i)", async () => {
    stubServer({
      providers: [
        { provider: "custom", models: ["my-model"], has_key: true, env_var: "OPENAI_API_KEY", requires_endpoint: true },
      ],
    });
    renderPage();
    await waitFor(() => expect(screen.getByTestId("run-blocked-reason")).toBeInTheDocument());
    fireEvent.change(template(), { target: { value: "hi" } });
    expect(runButton()).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /Connection/ }));
    fireEvent.change(screen.getByLabelText("Endpoint"), {
      target: { value: "http://localhost:8000/v1" },
    });
    expect(runButton()).toBeEnabled();
    expect(screen.queryByTestId("run-blocked-reason")).toBeNull();
  });

  it("lets a provider with no server key be picked, and runs it with a token", async () => {
    // It was a disabled option, so a gateway token could never be used with it.
    // jsdom has no scrollIntoView, which the Radix select calls on open.
    Element.prototype.scrollIntoView ??= () => {};
    const bodies = stubServer({
      providers: [
        { provider: "ollama", models: ["llama3.2"], has_key: true, env_var: null, requires_endpoint: false },
        { ...OPENAI, has_key: false },
      ],
    });
    renderPage();
    const trigger = await screen.findByRole("combobox", { name: "Provider" });
    await waitFor(() => expect(trigger).toHaveTextContent("ollama"));
    fireEvent.keyDown(trigger, { key: "ArrowDown" });
    const option = await screen.findByRole("option", { name: /openai/ });
    expect(option).not.toHaveAttribute("data-disabled");
    fireEvent.click(option);
    await waitFor(() => expect(trigger).toHaveTextContent("openai"));

    fireEvent.change(template(), { target: { value: "hi" } });
    expect(runButton()).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Token"), { target: { value: "gw-token" } });
    expect(runButton()).toBeEnabled();
    fireEvent.click(runButton());
    await waitFor(() => expect(bodies["/api/playground/stream"]).toHaveLength(1));
    expect(bodies["/api/playground/stream"][0]).toMatchObject({ provider: "openai", api_key: "gw-token" });
  });
});
