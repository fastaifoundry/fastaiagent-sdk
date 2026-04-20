import { Outlet } from "react-router-dom";
import { Toaster } from "sonner";
import { ErrorBoundary } from "@/components/shared/ErrorBoundary";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";

export function AppLayout() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-auto p-6">
          <ErrorBoundary>
            <Outlet />
          </ErrorBoundary>
        </main>
      </div>
      <Toaster position="bottom-right" richColors closeButton />
    </div>
  );
}
