/**
 * The trace note panel (1.89.0). The notes API had no UI; noted traces are
 * what `curate_from_traces(filter="noted")` reads. fetch is the only mock.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { NOTE_MAX_CHARS, TraceNotesCard } from "./TraceNotesCard";

function stub() {
  const calls: { method: string; url: string; body?: unknown }[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    const method = (init?.method ?? "GET").toUpperCase();
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ method, url, body });
    const payload =
      method === "DELETE"
        ? { deleted: 1 }
        : { status: "ok", note: body?.note ? { text: body.note, updated_at: "2026-10-14T00:00:00Z" } : null };
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });
  return calls;
}

afterEach(() => vi.restoreAllMocks());

describe("TraceNotesCard", () => {
  it("saves a note through POST /notes", async () => {
    const calls = stub();
    renderWithProviders(<TraceNotesCard traceId="abc" note={null} />);
    const box = screen.getByLabelText("Trace note");
    const save = screen.getByRole("button", { name: "Save note" });
    expect(save).toBeDisabled();
    fireEvent.change(box, { target: { value: "wrong refund amount" } });
    expect(screen.getByTestId("note-counter")).toHaveTextContent("19 / 20,000");
    fireEvent.click(save);
    await waitFor(() =>
      expect(calls.find((c) => c.method === "POST")).toMatchObject({
        url: "/api/traces/abc/notes",
        body: { note: "wrong refund amount" },
      })
    );
  });

  it("shows an existing note and clears it with DELETE", async () => {
    const calls = stub();
    renderWithProviders(
      <TraceNotesCard traceId="abc" note={{ text: "keep this", updated_at: "2026-10-14T00:00:00Z" }} />
    );
    expect(screen.getByLabelText("Trace note")).toHaveValue("keep this");
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    await waitFor(() =>
      expect(calls.find((c) => c.method === "DELETE")?.url).toBe("/api/traces/abc/notes")
    );
    await waitFor(() => expect(screen.getByLabelText("Trace note")).toHaveValue(""));
  });

  it("has no Clear without a note, and caps the text at the server's limit", () => {
    stub();
    renderWithProviders(<TraceNotesCard traceId="abc" note={null} />);
    expect(screen.queryByRole("button", { name: "Clear" })).toBeNull();
    fireEvent.change(screen.getByLabelText("Trace note"), {
      target: { value: "x".repeat(NOTE_MAX_CHARS + 50) },
    });
    expect((screen.getByLabelText("Trace note") as HTMLTextAreaElement).value).toHaveLength(
      NOTE_MAX_CHARS
    );
  });
});
