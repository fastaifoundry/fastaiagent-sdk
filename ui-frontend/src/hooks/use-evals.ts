import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  EvalCaseFilters,
  EvalCompareResponse,
  EvalRunDetail,
  EvalRunsPage,
  EvalTrendPoint,
} from "@/lib/types";

interface Filters {
  dataset?: string | null;
  agent?: string | null;
  page?: number;
  page_size?: number;
}

function buildQuery(f: Filters): string {
  const params = new URLSearchParams();
  if (f.dataset) params.set("dataset", f.dataset);
  if (f.agent) params.set("agent", f.agent);
  params.set("page", String(f.page ?? 1));
  params.set("page_size", String(f.page_size ?? 100));
  return params.toString() ? `?${params.toString()}` : "";
}

export function useEvalRuns(filters: Filters = {}) {
  return useQuery({
    queryKey: ["evals", filters],
    queryFn: () => api.get<EvalRunsPage>(`/evals${buildQuery(filters)}`),
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

export function useEvalTrend(dataset?: string | null) {
  return useQuery({
    queryKey: ["eval-trend", dataset ?? null],
    queryFn: () => {
      const q = dataset ? `?dataset=${encodeURIComponent(dataset)}` : "";
      return api.get<{ points: EvalTrendPoint[] }>(`/evals/trend${q}`);
    },
  });
}
