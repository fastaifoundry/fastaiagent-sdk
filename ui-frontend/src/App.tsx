import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "@/components/theme/ThemeProvider";
import { AppLayout } from "@/components/layout/AppLayout";
import { LoginPage } from "@/pages/LoginPage";
import { OverviewPage } from "@/pages/OverviewPage";
import { useAuth } from "@/hooks/use-auth";
import { useAuthStore } from "@/stores/auth-store";

// Route-level code-splitting keeps the initial bundle tight. Heavy deps
// (recharts, the replay flow) only load when the user navigates there.
const TracesPage = lazy(() =>
  import("@/pages/TracesPage").then((m) => ({ default: m.TracesPage }))
);
const TraceDetailPage = lazy(() =>
  import("@/pages/TraceDetailPage").then((m) => ({ default: m.TraceDetailPage }))
);
const AgentReplayPage = lazy(() =>
  import("@/pages/AgentReplayPage").then((m) => ({ default: m.AgentReplayPage }))
);
const EvalRunsPage = lazy(() =>
  import("@/pages/EvalRunsPage").then((m) => ({ default: m.EvalRunsPage }))
);
const EvalRunDetailPage = lazy(() =>
  import("@/pages/EvalRunDetailPage").then((m) => ({
    default: m.EvalRunDetailPage,
  }))
);
const EvalComparePage = lazy(() =>
  import("@/pages/EvalComparePage").then((m) => ({
    default: m.EvalComparePage,
  }))
);
const PromptsPage = lazy(() =>
  import("@/pages/PromptsPage").then((m) => ({ default: m.PromptsPage }))
);
const PromptEditorPage = lazy(() =>
  import("@/pages/PromptEditorPage").then((m) => ({ default: m.PromptEditorPage }))
);
const PlaygroundPage = lazy(() =>
  import("@/pages/PlaygroundPage").then((m) => ({ default: m.PlaygroundPage }))
);
const GuardrailsPage = lazy(() =>
  import("@/pages/GuardrailsPage").then((m) => ({ default: m.GuardrailsPage }))
);
const GuardrailEventDetailPage = lazy(() =>
  import("@/pages/GuardrailEventDetailPage").then((m) => ({
    default: m.GuardrailEventDetailPage,
  }))
);
const AgentsPage = lazy(() =>
  import("@/pages/AgentsPage").then((m) => ({ default: m.AgentsPage }))
);
const AgentDetailPage = lazy(() =>
  import("@/pages/AgentDetailPage").then((m) => ({ default: m.AgentDetailPage }))
);
const TraceComparePage = lazy(() =>
  import("@/pages/TraceComparePage").then((m) => ({ default: m.TraceComparePage }))
);
const DatasetsPage = lazy(() =>
  import("@/pages/DatasetsPage").then((m) => ({ default: m.DatasetsPage }))
);
const DatasetDetailPage = lazy(() =>
  import("@/pages/DatasetDetailPage").then((m) => ({
    default: m.DatasetDetailPage,
  }))
);
const AnalyticsPage = lazy(() =>
  import("@/pages/AnalyticsPage").then((m) => ({ default: m.AnalyticsPage }))
);
const ThreadPage = lazy(() =>
  import("@/pages/ThreadPage").then((m) => ({ default: m.ThreadPage }))
);
const KbListPage = lazy(() =>
  import("@/pages/KbListPage").then((m) => ({ default: m.KbListPage }))
);
const KbDetailPage = lazy(() =>
  import("@/pages/KbDetailPage").then((m) => ({ default: m.KbDetailPage }))
);
const WorkflowsPage = lazy(() =>
  import("@/pages/WorkflowsPage").then((m) => ({ default: m.WorkflowsPage }))
);
const WorkflowDetailPage = lazy(() =>
  import("@/pages/WorkflowDetailPage").then((m) => ({
    default: m.WorkflowDetailPage,
  }))
);
const ApprovalsPage = lazy(() =>
  import("@/pages/ApprovalsPage").then((m) => ({ default: m.ApprovalsPage }))
);
const ApprovalDetailPage = lazy(() =>
  import("@/pages/ApprovalDetailPage").then((m) => ({
    default: m.ApprovalDetailPage,
  }))
);
const ExecutionPage = lazy(() =>
  import("@/pages/ExecutionPage").then((m) => ({ default: m.ExecutionPage }))
);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 30_000,
    },
  },
});

function AuthGuard({ children }: { children: React.ReactNode }) {
  const { data, isLoading } = useAuth();
  // Fall back to the persisted zustand store while the first auth check is
  // in flight — avoids flashing /login on a hard refresh.
  const persisted = useAuthStore();

  if (isLoading && !persisted.authenticated && !persisted.noAuth) {
    return (
      <div className="flex h-screen items-center justify-center text-sm text-muted-foreground">
        Loading…
      </div>
    );
  }

  const authed = data?.authenticated ?? persisted.authenticated;
  const noAuth = data?.no_auth ?? persisted.noAuth;

  if (!authed && !noAuth) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

function PageFallback() {
  return (
    <div className="flex items-center justify-center py-12 text-sm text-muted-foreground">
      <span className="animate-pulse">Loading…</span>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route
              element={
                <AuthGuard>
                  <AppLayout />
                </AuthGuard>
              }
            >
              <Route path="/" element={<OverviewPage />} />
              <Route
                path="/analytics"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <AnalyticsPage />
                  </Suspense>
                }
              />
              <Route
                path="/threads/:threadId"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <ThreadPage />
                  </Suspense>
                }
              />
              <Route
                path="/traces"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <TracesPage />
                  </Suspense>
                }
              />
              <Route
                path="/traces/compare"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <TraceComparePage />
                  </Suspense>
                }
              />
              <Route
                path="/traces/:traceId"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <TraceDetailPage />
                  </Suspense>
                }
              />
              <Route
                path="/traces/:traceId/replay"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <AgentReplayPage />
                  </Suspense>
                }
              />
              <Route
                path="/evals"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <EvalRunsPage />
                  </Suspense>
                }
              />
              <Route
                path="/evals/compare"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <EvalComparePage />
                  </Suspense>
                }
              />
              <Route
                path="/evals/:runId"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <EvalRunDetailPage />
                  </Suspense>
                }
              />
              <Route
                path="/datasets"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <DatasetsPage />
                  </Suspense>
                }
              />
              <Route
                path="/datasets/:name"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <DatasetDetailPage />
                  </Suspense>
                }
              />
              <Route
                path="/prompts"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <PromptsPage />
                  </Suspense>
                }
              />
              <Route
                path="/prompts/:slug"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <PromptEditorPage />
                  </Suspense>
                }
              />
              <Route
                path="/playground"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <PlaygroundPage />
                  </Suspense>
                }
              />
              <Route
                path="/guardrails"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <GuardrailsPage />
                  </Suspense>
                }
              />
              <Route
                path="/guardrail-events/:eventId"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <GuardrailEventDetailPage />
                  </Suspense>
                }
              />
              <Route
                path="/agents"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <AgentsPage />
                  </Suspense>
                }
              />
              <Route
                path="/agents/:name"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <AgentDetailPage />
                  </Suspense>
                }
              />
              <Route
                path="/kb"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <KbListPage />
                  </Suspense>
                }
              />
              <Route
                path="/kb/:name"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <KbDetailPage />
                  </Suspense>
                }
              />
              <Route
                path="/workflows"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <WorkflowsPage />
                  </Suspense>
                }
              />
              <Route
                path="/workflows/:runnerType/:name"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <WorkflowDetailPage />
                  </Suspense>
                }
              />
              <Route
                path="/approvals"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <ApprovalsPage />
                  </Suspense>
                }
              />
              <Route
                path="/approvals/:execution_id"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <ApprovalDetailPage />
                  </Suspense>
                }
              />
              <Route
                path="/executions/:execution_id"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <ExecutionPage />
                  </Suspense>
                }
              />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
