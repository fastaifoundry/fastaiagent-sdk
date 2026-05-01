import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { WorkflowTopology } from "@/lib/types";

export function useWorkflowTopology(
  runnerType: string | undefined,
  name: string | undefined
) {
  return useQuery({
    queryKey: ["workflow-topology", runnerType, name],
    queryFn: () =>
      api.get<WorkflowTopology>(
        `/workflows/${encodeURIComponent(runnerType!)}/${encodeURIComponent(name!)}/topology`
      ),
    enabled: !!runnerType && !!name,
    retry: false,
  });
}
