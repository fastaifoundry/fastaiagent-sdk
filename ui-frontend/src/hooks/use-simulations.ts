import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  SimulationCaseFilters,
  SimulationRunDetail,
  SimulationRunsPage,
} from "@/lib/types";

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

export function useSimulationRuns(filters: Filters = {}) {
  return useQuery({
    queryKey: ["simulations", filters],
    queryFn: () =>
      api.get<SimulationRunsPage>(`/simulations${buildQuery(filters)}`),
  });
}

export function useSimulationRun(
  runId: string | undefined,
  filters: SimulationCaseFilters = {}
) {
  return useQuery({
    queryKey: ["simulation", runId, filters],
    queryFn: () => {
      const p = new URLSearchParams();
      if (filters.outcome) p.set("outcome", filters.outcome);
      const qs = p.toString() ? `?${p.toString()}` : "";
      return api.get<SimulationRunDetail>(`/simulations/${runId}${qs}`);
    },
    enabled: !!runId,
  });
}
