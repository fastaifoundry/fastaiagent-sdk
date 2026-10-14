import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useIncludeExperiments } from "@/stores/preferences-store";
import type {
  AgentDependencies,
  AgentSummary,
  AgentToolsResponse,
} from "@/lib/types";

// Agent stats leave out experiment traces (a replay rerun is not a run of
// the agent) unless the user opted in (1.89.0).
export function useAgents() {
  const includeExperiments = useIncludeExperiments();
  return useQuery({
    queryKey: ["agents", includeExperiments],
    queryFn: () =>
      api.get<{ agents: AgentSummary[] }>(
        `/agents?include_experiments=${includeExperiments}`
      ),
  });
}

export function useAgent(name: string | undefined) {
  const includeExperiments = useIncludeExperiments();
  return useQuery({
    queryKey: ["agent", name, includeExperiments],
    queryFn: () =>
      api.get<AgentSummary>(
        `/agents/${encodeURIComponent(name!)}?include_experiments=${includeExperiments}`
      ),
    enabled: !!name,
  });
}

export function useAgentTools(name: string | undefined) {
  return useQuery({
    queryKey: ["agent-tools", name],
    queryFn: () =>
      api.get<AgentToolsResponse>(
        `/agents/${encodeURIComponent(name!)}/tools`
      ),
    enabled: !!name,
  });
}

export function useAgentDependencies(name: string | undefined) {
  return useQuery({
    queryKey: ["agent-dependencies", name],
    queryFn: () =>
      api.get<AgentDependencies>(
        `/agents/${encodeURIComponent(name!)}/dependencies`
      ),
    enabled: !!name,
  });
}
