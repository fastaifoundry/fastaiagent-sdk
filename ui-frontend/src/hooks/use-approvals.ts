import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

export interface PendingInterrupt {
  execution_id: string;
  chain_name: string;
  node_id: string;
  reason: string;
  context: Record<string, unknown>;
  agent_path: string | null;
  created_at: string;
}

interface PendingPayload {
  count: number;
  items: PendingInterrupt[];
}

/** Polls only on focus / mount — refresh-based reads (memory rule). */
export function usePendingInterrupts(limit = 100) {
  return useQuery({
    queryKey: ["pending-interrupts", limit],
    queryFn: () =>
      api.get<PendingPayload>(`/pending-interrupts?limit=${limit}`),
  });
}

export interface ResumeBody {
  approved: boolean;
  metadata?: Record<string, unknown>;
  reason?: string;
}

export interface ResumeResponse {
  execution_id: string;
  chain_name: string;
  reason: string | null;
  /** 1.89.0: the signed-in user when auth is on and none was given. */
  resolver?: string | null;
  /** 1.89.0: the resumed run's own trace (one per resume). */
  trace_id?: string | null;
  result: {
    status: "completed" | "paused";
    output?: string;
    pending_interrupt?: Record<string, unknown> | null;
    [key: string]: unknown;
  };
}

export function useResumeExecution(executionId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ResumeBody) =>
      api.post<ResumeResponse>(`/executions/${executionId}/resume`, body),
    // The pause is gone and a new trace exists: every view of it is stale
    // (1.89.0; the approvals list kept showing a resolved pause).
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["execution", executionId] });
      qc.invalidateQueries({ queryKey: ["pending-interrupts"] });
      qc.invalidateQueries({ queryKey: ["pending-for-execution", executionId] });
      qc.invalidateQueries({ queryKey: ["traces"] });
    },
  });
}
