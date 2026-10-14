/**
 * REST client for the FastAIAgent Local UI.
 *
 * Backend lives on the same origin as the SPA (served by FastAPI),
 * so base is just "/api". Session cookies are auto-sent by the browser
 * because `credentials: "include"` is set on every request.
 */

import { normalizeErrorBody, type ErrorInfo } from "@/lib/api-error";

export class ApiError extends Error {
  status: number;
  body?: unknown;
  /** The normalised body: field errors, correlation id, provider details. */
  info: ErrorInfo;

  constructor(status: number, message: string, body?: unknown, info?: ErrorInfo) {
    // Always a string, whatever shape `detail` had (1.89.0, P1).
    const resolved = info ?? normalizeErrorBody(body, status, message);
    super(typeof message === "string" && message ? message : resolved.message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
    this.info = resolved;
  }
}

const BASE = "/api";

type Method = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

interface RequestOptions {
  body?: unknown;
  signal?: AbortSignal;
}

const SAFE_METHODS = new Set<Method>(["GET"]);

/**
 * security_review_1.md M4 — read the double-submit CSRF cookie.
 *
 * The backend issues ``fastaiagent_csrf`` (NOT httpOnly) on safe
 * responses; we echo it back as ``X-CSRF-Token`` on every state-changing
 * call. Returns null if the cookie isn't set yet — the very first
 * mutating call after a fresh page load may need a no-op GET first.
 */
export function readCsrfCookie(): string | null {
  const match = document.cookie.match(/(?:^|; )fastaiagent_csrf=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
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
  const headers: Record<string, string> = {};
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  if (!SAFE_METHODS.has(method)) {
    const csrf = readCsrfCookie();
    if (csrf) {
      headers["X-CSRF-Token"] = csrf;
    }
  }
  if (Object.keys(headers).length > 0) {
    init.headers = headers;
  }

  const res = await fetch(`${BASE}${path}`, init);

  if (!res.ok) {
    let payload: unknown = null;
    try {
      payload = await res.json();
    } catch {
      /* swallow */
    }
    // `detail` is a string, a list (FastAPI 422) or an object; `new Error()`
    // turned the last two into "[object Object]".
    const info = normalizeErrorBody(payload, res.status, res.statusText);
    throw new ApiError(res.status, info.message, payload, info);
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
