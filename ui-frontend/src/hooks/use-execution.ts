import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { PendingInterrupt } from "@/hooks/use-approvals";

export interface ExecutionCheckpoint {
  checkpoint_id: string;
  parent_checkpoint_id: string | null;
  chain_name: string;
  execution_id: string;
  node_id: string;
  node_index: number;
  status: string;
  state_snapshot: Record<string, unknown>;
  node_input: Record<string, unknown>;
  node_output: Record<string, unknown>;
  iteration: number;
  iteration_counters: Record<string, number>;
  interrupt_reason: string | null;
  interrupt_context: Record<string, unknown>;
  agent_path: string | null;
  created_at: string;
}

export interface ExecutionDetail {
  execution_id: string;
  chain_name: string;
  status: string;
  agent_path: string | null;
  checkpoint_count: number;
  latest_checkpoint_id: string;
  latest_state_snapshot: Record<string, unknown>;
  checkpoints: ExecutionCheckpoint[];
}

export function useExecution(executionId: string | undefined) {
  return useQuery({
    queryKey: ["execution", executionId],
    queryFn: () => api.get<ExecutionDetail>(`/executions/${executionId}`),
    enabled: !!executionId,
  });
}

/** Find the single pending interrupt for one execution, if any.
 *
 * The /api/pending-interrupts endpoint is small; filtering client-side
 * is cheaper than adding a per-execution endpoint.
 */
export function usePendingForExecution(executionId: string | undefined) {
  return useQuery({
    queryKey: ["pending-for-execution", executionId],
    queryFn: async (): Promise<PendingInterrupt | null> => {
      const body = await api.get<{ items: PendingInterrupt[] }>(
        "/pending-interrupts"
      );
      return body.items.find((p) => p.execution_id === executionId) ?? null;
    },
    enabled: !!executionId,
  });
}
