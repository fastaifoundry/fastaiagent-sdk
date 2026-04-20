import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { AnalyticsPayload, TraceScores, ThreadDetail } from "@/lib/types";

export function useAnalytics(windowHours: number = 168, granularity: "hour" | "day" = "hour") {
  return useQuery({
    queryKey: ["analytics", windowHours, granularity],
    queryFn: () =>
      api.get<AnalyticsPayload>(
        `/analytics?hours=${windowHours}&granularity=${granularity}`
      ),
  });
}

export function useTraceScores(traceId: string | undefined) {
  return useQuery({
    queryKey: ["trace-scores", traceId],
    queryFn: () => api.get<TraceScores>(`/traces/${traceId}/scores`),
    enabled: !!traceId,
  });
}

export function useThread(threadId: string | undefined) {
  return useQuery({
    queryKey: ["thread", threadId],
    queryFn: () => api.get<ThreadDetail>(`/threads/${threadId}`),
    enabled: !!threadId,
  });
}
