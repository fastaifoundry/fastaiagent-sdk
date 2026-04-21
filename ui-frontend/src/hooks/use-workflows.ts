import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { WorkflowListResponse, WorkflowSummary } from "@/lib/types";

export function useWorkflows(runnerType?: string | null) {
  const qs = runnerType ? `?runner_type=${encodeURIComponent(runnerType)}` : "";
  return useQuery({
    queryKey: ["workflows", runnerType ?? "all"],
    queryFn: () => api.get<WorkflowListResponse>(`/workflows${qs}`),
  });
}

export function useWorkflow(
  runnerType: string | undefined,
  name: string | undefined
) {
  return useQuery({
    queryKey: ["workflow", runnerType, name],
    queryFn: () =>
      api.get<WorkflowSummary>(
        `/workflows/${encodeURIComponent(runnerType!)}/${encodeURIComponent(name!)}`
      ),
    enabled: !!runnerType && !!name,
  });
}
