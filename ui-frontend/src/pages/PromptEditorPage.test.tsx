import { afterEach, describe, expect, it, vi } from "vitest";
import { Routes, Route } from "react-router-dom";
import { fireEvent, screen, waitFor } from "@testing-library/react";
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
