/**
 * REST client for the FastAIAgent Local UI.
 *
 * Backend lives on the same origin as the SPA (served by FastAPI),
 * so base is just "/api". Session cookies are auto-sent by the browser
 * because `credentials: "include"` is set on every request.
 */

export class ApiError extends Error {
  status: number;
  body?: unknown;

  constructor(status: number, message: string, body?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

const BASE = "/api";

type Method = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

interface RequestOptions {
  body?: unknown;
  signal?: AbortSignal;
}

async function request<T>(
  method: Method,
  path: string,
  { body, signal }: RequestOptions = {}
): Promise<T> {
  const init: RequestInit = {
    method,
    credentials: "include",
    signal,
  };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }

  const res = await fetch(`${BASE}${path}`, init);

  if (!res.ok) {
    let payload: unknown = null;
    try {
      payload = await res.json();
    } catch {
      /* swallow */
    }
    const detail =
      (payload as { detail?: string } | null)?.detail ?? res.statusText;
    throw new ApiError(res.status, detail, payload);
  }

  // 204 No Content
  if (res.status === 204) {
    return undefined as T;
  }

  const contentType = res.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return (await res.json()) as T;
  }
  return (await res.text()) as unknown as T;
}

export const api = {
  get: <T>(path: string, signal?: AbortSignal) => request<T>("GET", path, { signal }),
  post: <T>(path: string, body?: unknown, signal?: AbortSignal) =>
    request<T>("POST", path, { body, signal }),
  put: <T>(path: string, body?: unknown, signal?: AbortSignal) =>
    request<T>("PUT", path, { body, signal }),
  patch: <T>(path: string, body?: unknown, signal?: AbortSignal) =>
    request<T>("PATCH", path, { body, signal }),
  delete: <T = void>(path: string, signal?: AbortSignal) =>
    request<T>("DELETE", path, { signal }),
};
