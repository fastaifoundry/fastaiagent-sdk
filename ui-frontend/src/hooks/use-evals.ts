import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  EvalCaseFilters,
  EvalCompareResponse,
  EvalRunDetail,
  EvalRunFilters,
  EvalRunsPage,
  EvalTrendPoint,
} from "@/lib/types";

/** Rows per page on the eval runs list. */
export const EVAL_RUNS_PAGE_SIZE = 50;

function buildQuery(f: EvalRunFilters): string {
  const params = new URLSearchParams();
  if (f.dataset) params.set("dataset", f.dataset);
  if (f.agent) params.set("agent", f.agent);
  if (f.kind) params.set("kind", f.kind);
  if (f.exclude_kind) params.set("exclude_kind", f.exclude_kind);
  if (f.q) params.set("q", f.q);
  if (f.gate) params.set("gate", f.gate);
  params.set("page", String(f.page ?? 1));
  params.set("page_size", String(f.page_size ?? 100));
  return params.toString() ? `?${params.toString()}` : "";
}

export function useEvalRuns(filters: EvalRunFilters = {}) {
  return useQuery({
    queryKey: ["evals", filters],
    queryFn: () => api.get<EvalRunsPage>(`/evals${buildQuery(filters)}`),
    // Keep the current page on screen while the next one loads.
    placeholderData: keepPreviousData,
  });
}

export function useEvalRun(
  runId: string | undefined,
  filters: EvalCaseFilters = {}
) {
  return useQuery({
    queryKey: ["eval", runId, filters],
    queryFn: () => {
      const p = new URLSearchParams();
      if (filters.scorer) p.set("scorer", filters.scorer);
      if (filters.outcome) p.set("outcome", filters.outcome);
      if (filters.q) p.set("q", filters.q);
      const qs = p.toString() ? `?${p.toString()}` : "";
      return api.get<EvalRunDetail>(`/evals/${runId}${qs}`);
    },
    enabled: !!runId,
    // Case filters change often (typing in the search box) — keep the
    // data snappy but avoid hammering the server; useQuery's default
    // debounce-on-rerender is already tight enough, so no extra logic.
  });
}

export function useEvalCompare(a: string | undefined, b: string | undefined) {
  return useQuery({
    queryKey: ["eval-compare", a, b],
    queryFn: () =>
      api.get<EvalCompareResponse>(
        `/evals/compare?a=${encodeURIComponent(a!)}&b=${encodeURIComponent(b!)}`
      ),
    enabled: !!a && !!b,
  });
}

/**
 * Quality over time. Dataset checks (ui-echo) and AutoLLM candidate runs are
 * left out unless `includeAll` (the API's `include=all`).
 */
export function useEvalTrend(dataset?: string | null, includeAll = false) {
  return useQuery({
    queryKey: ["eval-trend", dataset ?? null, includeAll],
    queryFn: () => {
      const p = new URLSearchParams();
      if (dataset) p.set("dataset", dataset);
      if (includeAll) p.set("include", "all");
      const q = p.toString() ? `?${p.toString()}` : "";
      return api.get<{ points: EvalTrendPoint[] }>(`/evals/trend${q}`);
    },
    placeholderData: keepPreviousData,
  });
}
