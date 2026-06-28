import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { OptimizeRunDetail, OptimizeRunsPage } from "@/lib/types";

interface Filters {
  agent?: string | null;
  page?: number;
  page_size?: number;
}

function buildQuery(f: Filters): string {
  const params = new URLSearchParams();
  if (f.agent) params.set("agent", f.agent);
  params.set("page", String(f.page ?? 1));
  params.set("page_size", String(f.page_size ?? 100));
  return params.toString() ? `?${params.toString()}` : "";
}

export function useOptimizeRuns(filters: Filters = {}) {
  return useQuery({
    queryKey: ["optimizes", filters],
    queryFn: () => api.get<OptimizeRunsPage>(`/optimizes${buildQuery(filters)}`),
  });
}

export function useOptimizeRun(runId: string | undefined) {
  return useQuery({
    queryKey: ["optimize", runId],
    queryFn: () => api.get<OptimizeRunDetail>(`/optimizes/${runId}`),
    enabled: !!runId,
  });
}
