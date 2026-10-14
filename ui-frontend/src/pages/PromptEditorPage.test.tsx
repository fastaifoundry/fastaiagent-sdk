import { afterEach, describe, expect, it, vi } from "vitest";
import { Routes, Route } from "react-router-dom";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { PromptEditorPage } from "./PromptEditorPage";

/**
 * The editor edits the RAW template and never wipes metadata (P2, P3, 1.89.0).
 *
 * It used to seed the textarea from the resolved template, so saving replaced
 * {{@tone}} with a frozen copy of the fragment; and it sent `metadata: {}` on
 * every save. The api.ts boundary (fetch) is the only mock surface.
 */

const RAW = "{{@tone}} Hello {{name}}";

const v1 = {
  slug: "greet",
  version: 1,
  template: "Be warm. Hello {{name}}",
  template_raw: RAW,
  fragments: ["tone"],
  variables: ["name"],
  metadata: { owner: "ops" },
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function stubApi() {
  const puts: unknown[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    const method = (init?.method ?? "GET").toUpperCase();
    if (method === "PUT" && url.endsWith("/api/prompts/greet")) {
      puts.push(JSON.parse(String(init?.body)));
      return json({ ...v1, version: 2, template_raw: RAW + "!" });
    }
    if (url.includes("/api/prompts/greet/versions/")) return json(v1);
    if (url.endsWith("/api/prompts/greet/versions")) {
      return json({ versions: [{ ...v1, created_at: null, created_by: "code" }] });
    }
    if (url.endsWith("/api/prompts/greet/lineage")) {
      return json({ trace_ids: [], eval_run_ids: [] });
    }
    if (url.endsWith("/api/prompts/greet")) {
      return json({ ...v1, latest_version: 1, registry_is_local: true });
    }
    return json({});
  });
  return puts;
}

function renderEditor() {
  renderWithProviders(
    <Routes>
      <Route path="/prompts/:slug" element={<PromptEditorPage />} />
    </Routes>,
    { route: "/prompts/greet" }
  );
}

afterEach(() => vi.restoreAllMocks());

describe("PromptEditorPage", () => {
  it("edits the raw template, fragment references intact", async () => {
    stubApi();
    renderEditor();
    const textarea = (await screen.findByDisplayValue(RAW)) as HTMLTextAreaElement;
    expect(textarea.value).toBe(RAW);
    expect(screen.getByTestId("prompt-fragments")).toHaveTextContent("{{@tone}}");
  });

  it("saves the raw text with its base version and no metadata", async () => {
    const puts = stubApi();
    renderEditor();
    await screen.findByDisplayValue(RAW);
    // Let the selected version's detail land first: it re-seeds the draft.
    await waitFor(() =>
      expect(
        vi
          .mocked(globalThis.fetch)
          .mock.calls.some(([u]) => String(u).includes("/api/prompts/greet/versions/1"))
      ).toBe(true)
    );
    await new Promise((r) => setTimeout(r, 0));
    const textarea = screen.getByDisplayValue(RAW);
    fireEvent.change(textarea, { target: { value: RAW + "!" } });
    fireEvent.click(screen.getByRole("button", { name: /save as new version/i }));

    await waitFor(() => expect(puts).toHaveLength(1));
    const body = puts[0] as Record<string, unknown>;
    expect(body.template).toBe(RAW + "!");
    expect(body.base_version).toBe(1);
    expect(body).not.toHaveProperty("metadata");
  });
});

/**
 * Aliases and the compare panel (1.89.0, P10a/b). Versions 1 and 3 exist (not
 * contiguous); `production` points at v1.
 */
function stubAliasApi() {
  const calls: { method: string; url: string; body?: unknown }[] = [];
  const version = (v: number) => ({
    ...v1,
    version: v,
    template: `T${v}`,
    template_raw: `T${v}`,
    fragments: [],
  });
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    const method = (init?.method ?? "GET").toUpperCase();
    calls.push({ method, url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
    if (url.includes("/aliases/")) return json({ slug: "greet", aliases: { production: 1 } });
    if (url.includes("/api/prompts/greet/diff")) {
      return json({
        diff: "--- greet v1\n+++ greet v3\n@@ -1 +1 @@\n-T1\n+T3",
        a: { version: 1, template: "T1", metadata: {} },
        b: { version: 3, template: "T3", metadata: {} },
      });
    }
    const m = url.match(/\/versions\/(\d+)$/);
    if (m) return json(version(Number(m[1])));
    if (url.endsWith("/api/prompts/greet/versions")) {
      return json({
        versions: [
          { ...version(1), version: "1", created_at: null, created_by: "ui", aliases: ["production"] },
          { ...version(3), version: "3", created_at: null, created_by: "ui", aliases: [] },
        ],
      });
    }
    if (url.endsWith("/api/prompts/greet/lineage")) {
      return json({ trace_ids: [], eval_run_ids: [] });
    }
    if (url.endsWith("/api/prompts/greet")) {
      return json({
        ...version(3),
        latest_version: 3,
        registry_is_local: true,
        aliases: { production: 1 },
      });
    }
    return json({});
  });
  return calls;
}

function renderAt(route: string) {
  renderWithProviders(
    <Routes>
      <Route path="/prompts/:slug" element={<PromptEditorPage />} />
    </Routes>,
    { route }
  );
}

describe("PromptEditorPage aliases and compare", () => {
  it("opens ?alias= at its version and hands the alias to the Playground", async () => {
    stubAliasApi();
    renderAt("/prompts/greet?alias=production");
    expect(await screen.findByDisplayValue("T1")).toBeInTheDocument();
    expect(screen.getByText("(editing v1)")).toBeInTheDocument();
    expect(screen.getAllByTestId("version-alias-chip")[0]).toHaveTextContent("production");
    const link = screen.getByRole("link", { name: /test in playground/i });
    expect(link.getAttribute("href")).toBe("/playground?prompt=greet&alias=production");
  });

  it("points a new alias at the selected version", async () => {
    const calls = stubAliasApi();
    renderAt("/prompts/greet");
    await screen.findByText("(editing v3)");
    fireEvent.change(screen.getByLabelText("Alias name"), { target: { value: "staging" } });
    fireEvent.click(screen.getByRole("button", { name: "Point at v3" }));
    await waitFor(() =>
      expect(calls.find((c) => c.method === "PUT")).toMatchObject({
        url: "/api/prompts/greet/aliases/staging",
        body: { version: 3 },
      })
    );
  });

  it("refuses an alias name the server would reject", async () => {
    stubAliasApi();
    renderAt("/prompts/greet");
    await screen.findByText("(editing v3)");
    fireEvent.change(screen.getByLabelText("Alias name"), { target: { value: "bad name" } });
    expect(screen.getByRole("button", { name: "Point at v3" })).toBeDisabled();
  });

  it("removes an alias", async () => {
    const calls = stubAliasApi();
    renderAt("/prompts/greet");
    fireEvent.click(await screen.findByRole("button", { name: "Remove alias production" }));
    await waitFor(() =>
      expect(calls.find((c) => c.method === "DELETE")?.url).toBe(
        "/api/prompts/greet/aliases/production"
      )
    );
  });

  it("compares the selected version with the one before it", async () => {
    const calls = stubAliasApi();
    renderAt("/prompts/greet");
    await screen.findByText("(editing v3)");
    fireEvent.click(screen.getByRole("button", { name: /compare versions/i }));
    const diff = await screen.findByTestId("prompt-diff");
    expect(calls.some((c) => c.url.endsWith("/diff?a=1&b=3"))).toBe(true);
    expect(within(diff).getByText("+T3").className).toContain("text-fa-success");
    expect(within(diff).getByText("-T1").className).toContain("text-destructive");
    expect(screen.getByTestId("prompt-side-1")).toHaveTextContent("T1");
    expect(screen.getByTestId("prompt-side-3")).toHaveTextContent("T3");
  });
});
