import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
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

export function useEvalRun(runId: string | undefined) {
  return useQuery({
    queryKey: ["eval", runId],
    queryFn: () => api.get<EvalRunDetail>(`/evals/${runId}`),
    enabled: !!runId,
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
