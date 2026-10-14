import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useIncludeExperiments } from "@/stores/preferences-store";
import type {
  CostBreakdownResponse,
  CostGroupBy,
} from "@/lib/types";

interface Params {
  groupBy: CostGroupBy;
  period?: "1d" | "7d" | "30d" | "all";
  chainName?: string | null;
  agent?: string | null;
}

export function useCostBreakdown({
  groupBy,
  period = "7d",
  chainName,
  agent,
}: Params) {
  const includeExperiments = useIncludeExperiments();
  const qs = new URLSearchParams({ group_by: groupBy, period });
  if (chainName) qs.set("chain_name", chainName);
  if (agent) qs.set("agent", agent);
  qs.set("include_experiments", String(includeExperiments));
  return useQuery({
    queryKey: ["cost-breakdown", groupBy, period, chainName, agent, includeExperiments],
    queryFn: () =>
      api.get<CostBreakdownResponse>(`/analytics/costs?${qs.toString()}`),
    enabled: groupBy !== "node" || !!chainName,
  });
}
