import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  PlaygroundModelsResponse,
  PlaygroundRunRequest,
  PlaygroundRunResponse,
  PlaygroundStreamEvent,
  SaveAsEvalRequest,
  SaveAsEvalResponse,
} from "@/lib/types";

export function usePlaygroundModels() {
  return useQuery({
    queryKey: ["playground-models"],
    queryFn: () =>
      api.get<PlaygroundModelsResponse>("/playground/models"),
  });
}

export function usePlaygroundRun() {
  return useMutation({
    mutationFn: (body: PlaygroundRunRequest) =>
      api.post<PlaygroundRunResponse>("/playground/run", body),
  });
}

export function useSaveAsEval() {
  return useMutation({
    mutationFn: (body: SaveAsEvalRequest) =>
      api.post<SaveAsEvalResponse>("/playground/save-as-eval", body),
  });
}

/**
 * Consume the playground SSE stream as an async iterator.
 *
 * EventSource is GET-only, so we POST via `fetch()` and parse the
 * `text/event-stream` body ourselves. Each yielded value is a parsed
 * `PlaygroundStreamEvent`.
 *
 * The caller is expected to wrap this in a `try { for await ... }` and
 * pass an `AbortSignal` to cancel mid-stream (the "Stop" button).
 */
export async function* streamPlayground(
  body: PlaygroundRunRequest,
  signal?: AbortSignal,
): AsyncGenerator<PlaygroundStreamEvent, void, unknown> {
  const res = await fetch("/api/playground/stream", {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    let message = res.statusText;
    try {
      const payload = (await res.json()) as { detail?: string };
      message = payload.detail ?? message;
    } catch {
      /* swallow */
    }
    yield { event: "error", message };
    return;
  }

  if (!res.body) {
    yield { event: "error", message: "No response body" };
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE messages are separated by a blank line.
    let nl: number;
    while ((nl = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 2);
      const parsed = parseSseMessage(raw);
      if (parsed) yield parsed;
    }
  }
}

function parseSseMessage(raw: string): PlaygroundStreamEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  const data = dataLines.join("\n");
  if (!data) return null;
  try {
    const payload = JSON.parse(data);
    if (event === "token") {
      return { event: "token", text: String(payload.text ?? "") };
    }
    if (event === "done") {
      return { event: "done", metadata: payload.metadata };
    }
    if (event === "error") {
      return {
        event: "error",
        message: String(payload.message ?? "stream error"),
      };
    }
  } catch {
    return null;
  }
  return null;
}
