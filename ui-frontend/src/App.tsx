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
const PromptsPage = lazy(() =>
  import("@/pages/PromptsPage").then((m) => ({ default: m.PromptsPage }))
);
const PromptEditorPage = lazy(() =>
  import("@/pages/PromptEditorPage").then((m) => ({ default: m.PromptEditorPage }))
);
const GuardrailsPage = lazy(() =>
  import("@/pages/GuardrailsPage").then((m) => ({ default: m.GuardrailsPage }))
);
const AgentsPage = lazy(() =>
  import("@/pages/AgentsPage").then((m) => ({ default: m.AgentsPage }))
);
const AgentDetailPage = lazy(() =>
  import("@/pages/AgentDetailPage").then((m) => ({ default: m.AgentDetailPage }))
);
const CompareTracesPage = lazy(() =>
  import("@/pages/stubs").then((m) => ({ default: m.CompareTracesPage }))
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
                    <CompareTracesPage />
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
                path="/evals/:runId"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <EvalRunDetailPage />
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
                path="/guardrails"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <GuardrailsPage />
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
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
