import { useLocation } from "react-router-dom";
import { ChevronRight, Command, LogOut } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { ThemeSwitcher } from "@/components/theme/ThemeSwitcher";
import { SkinSwitcher } from "@/components/theme/SkinSwitcher";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/stores/auth-store";
import { locate } from "./nav-config";

/**
 * ContentHeader — top bar for the new shell. Left: the project/auth context
 * that classic showed, plus a breadcrumb derived from the nav config. Right:
 * ⌘K, skin, theme, logout.
 *
 * The `header-breadcrumb` and `project-id` test ids are load-bearing — two
 * Playwright specs assert them (example-sweep.spec.ts, sprint1.spec.ts). Keep
 * them on the same elements as the classic Header.
 */
export function ContentHeader({ onOpenPalette }: { onOpenPalette: () => void }) {
  const location = useLocation();
  const { username, noAuth, projectId } = useAuthStore();
  const queryClient = useQueryClient();

  const { group, section, item } = locate(location.pathname);
  const crumbSection = section ?? group?.label ?? "Overview";
  const crumbLabel = item?.label ?? "Detail";

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
    <header className="flex h-12 shrink-0 items-center gap-2 border-b border-border bg-card/60 px-4 backdrop-blur">
      <div
        className="flex items-center gap-1.5 text-sm text-muted-foreground"
        data-testid="header-breadcrumb"
      >
        <span className="font-mono text-[11px] uppercase tracking-wider">
          Local
        </span>
        {projectId ? (
          <>
            <ChevronRight className="h-3.5 w-3.5 text-muted-foreground/50" />
            <span
              className="font-mono text-foreground"
              data-testid="project-id"
            >
              {projectId}
            </span>
          </>
        ) : null}
      </div>

      {/* Location breadcrumb — hidden on narrow viewports, as in Enterprise. */}
      <div className="ml-2 hidden items-center gap-1.5 border-l border-border pl-3 text-sm lg:flex">
        <span className="text-muted-foreground">{crumbSection}</span>
        <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="font-medium">{crumbLabel}</span>
      </div>

      <div className="ml-auto flex items-center gap-2">
        <button
          type="button"
          onClick={onOpenPalette}
          className="hidden items-center gap-2 rounded-md border border-input bg-background px-2.5 py-1.5 text-xs text-muted-foreground hover:bg-muted sm:flex"
        >
          <Command className="h-3.5 w-3.5" /> Search
          <span className="rounded border border-border bg-muted px-1 font-mono text-[10px]">
            ⌘K
          </span>
        </button>

        <SkinSwitcher />
        <ThemeSwitcher />

        {!noAuth && (
          <>
            <div className="h-5 w-px bg-border" />
            <span className="hidden text-xs text-muted-foreground lg:inline">
              {username ?? "…"}
            </span>
            <Button
              variant="ghost"
              size="icon-sm"
              title="Log out"
              onClick={() => logout.mutate()}
              disabled={logout.isPending}
            >
              <LogOut className="h-3.5 w-3.5" />
              <span className="sr-only">Logout</span>
            </Button>
          </>
        )}
      </div>
    </header>
  );
}
