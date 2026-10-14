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
  /** Replication state, not run state — null on a pre-v19 local.db. */
  synced: boolean | null;
  /** Set only when the checkpoint will NEVER reach the plane: the reason it
   *  was refused. `synced` is true alongside it, because the outbox has
   *  stopped retrying — the two together mean "we gave up on this one". */
  sync_error: string | null;
  /**
   * 1.89.0, on `interrupted` rows only: "pending" while the pause still waits,
   * "resolved" once someone resumed it. Absent from an older server.
   */
  interrupt_state?: "pending" | "resolved" | null;
}

/** How a resumed run was resolved, read from its trace (1.89.0). */
export interface ExecutionResume {
  node_id: string | null;
  kind: string | null;
  approved: boolean | null;
  resolver: string | null;
  reason: string | null;
}

/** One trace that ran this execution: the run, each resume, a fork. */
export interface ExecutionRun {
  trace_id: string;
  role: "run" | "resume" | "fork";
  name: string | null;
  started_at: string | null;
  ended_at: string | null;
  status: "completed" | "paused" | "failed" | string;
  /** Up to 2000 characters. */
  output: unknown;
  resume: ExecutionResume | null;
}

/** Who resolved the pause, and how. `source` says where it was read from. */
export interface ExecutionResolution {
  approved: boolean | null;
  resolver: string | null;
  reason: string | null;
  node_id?: string | null;
  at: string | null;
  trace_id: string | null;
  source: "trace" | "hitl_events" | string;
}

export interface ExecutionDetail {
  execution_id: string;
  chain_name: string;
  status: string;
  agent_path: string | null;
  checkpoint_count: number;
  latest_checkpoint_id: string;
  latest_state_snapshot: Record<string, unknown>;
  /** >0 means the plane's copy of this run is incomplete. */
  quarantined_count?: number;
  checkpoints: ExecutionCheckpoint[];
  // 1.89.0: what happened, not only what is stored.
  pending_interrupt?: PendingInterrupt | null;
  runs?: ExecutionRun[];
  resolution?: ExecutionResolution | null;
  final_output?: unknown;
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
