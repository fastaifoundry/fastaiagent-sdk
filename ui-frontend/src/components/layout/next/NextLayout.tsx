import { useEffect, useState } from "react";
import { Outlet } from "react-router-dom";
import { Toaster } from "sonner";
import { ErrorBoundary } from "@/components/shared/ErrorBoundary";
import { FlyoutNav } from "./FlyoutNav";
import { ContentHeader } from "./ContentHeader";
import { CommandPalette } from "./CommandPalette";

/**
 * NextLayout — the new shell: icon rail + content header + routed body.
 *
 * Mirrors classic AppLayout's responsibilities exactly (error boundary,
 * toaster, outlet) so switching skins changes appearance and navigation
 * only, never behaviour.
 *
 * `<main>` keeps `p-6`, unlike the Enterprise shell where each page owns its
 * own frame: dropping it here would strip padding from all 33 pages at once.
 * Pages opt into the centred /next frame individually instead.
 */
export function NextLayout() {
  const [paletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden">
      <FlyoutNav onOpenPalette={() => setPaletteOpen(true)} />
      <div className="flex flex-1 flex-col overflow-hidden">
        <ContentHeader onOpenPalette={() => setPaletteOpen(true)} />
        <main className="flex-1 overflow-auto p-6">
          <ErrorBoundary>
            <Outlet />
          </ErrorBoundary>
        </main>
      </div>
      <Toaster position="bottom-right" richColors closeButton />
      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
      />
    </div>
  );
}
