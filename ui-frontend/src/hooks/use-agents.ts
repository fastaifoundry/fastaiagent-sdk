import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { AgentSummary, AgentToolsResponse } from "@/lib/types";

export function useAgents() {
  return useQuery({
    queryKey: ["agents"],
    queryFn: () => api.get<{ agents: AgentSummary[] }>("/agents"),
  });
}

export function useAgent(name: string | undefined) {
  return useQuery({
    queryKey: ["agent", name],
    queryFn: () => api.get<AgentSummary>(`/agents/${encodeURIComponent(name!)}`),
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
