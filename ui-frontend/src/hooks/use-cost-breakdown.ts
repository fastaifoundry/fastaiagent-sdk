import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
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
  const qs = new URLSearchParams({ group_by: groupBy, period });
  if (chainName) qs.set("chain_name", chainName);
  if (agent) qs.set("agent", agent);
  return useQuery({
    queryKey: ["cost-breakdown", groupBy, period, chainName, agent],
    queryFn: () =>
      api.get<CostBreakdownResponse>(`/analytics/costs?${qs.toString()}`),
    enabled: groupBy !== "node" || !!chainName,
  });
}
