import { LogOut } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { ThemeSwitcher } from "@/components/theme/ThemeSwitcher";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/stores/auth-store";

export function Header() {
  const { username, noAuth, projectId } = useAuthStore();
  const queryClient = useQueryClient();

  const logout = useMutation({
    mutationFn: () => api.post<void>("/auth/logout"),
    onSuccess: () => {
      useAuthStore.getState().clear();
      queryClient.invalidateQueries({ queryKey: ["auth", "status"] });
      toast.success("Logged out");
    },
    onError: (e) => {
      if (e instanceof ApiError) toast.error(e.message);
      else toast.error("Logout failed");
    },
  });

  return (
    <header className="flex h-14 items-center justify-between border-b bg-card px-6">
      <div className="text-sm text-muted-foreground" data-testid="header-breadcrumb">
        <span className="font-mono">Local UI</span>
        {projectId ? (
          <>
            <span className="mx-2 text-border">//</span>
            <span className="font-mono text-foreground" data-testid="project-id">
              {projectId}
            </span>
          </>
        ) : null}
        <span className="mx-2 text-border">//</span>
        <span>{noAuth ? "auth disabled" : `signed in as ${username ?? "…"}`}</span>
      </div>
      <div className="flex items-center gap-2">
        <ThemeSwitcher />
        {!noAuth && (
          <>
            <div className="h-5 w-px bg-border" />
            <Button
              variant="ghost"
              size="sm"
              onClick={() => logout.mutate()}
              disabled={logout.isPending}
            >
              <LogOut className="mr-1.5 h-3.5 w-3.5" />
              Logout
            </Button>
          </>
        )}
      </div>
    </header>
  );
}
