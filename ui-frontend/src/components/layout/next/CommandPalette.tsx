import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Command, CornerDownLeft, Search } from "lucide-react";
import { ALL_NAV_ITEMS } from "./nav-config";

/**
 * CommandPalette — ⌘K overlay for jumping between pages.
 *
 * Navigation only: it lists the same destinations the rail offers, so every
 * link stays keyboard-reachable while the flyout is closed. Enterprise's
 * palette also carries "quick actions" (create agent, invite teammate); those
 * are create-flows the Local UI does not have, so they are omitted rather
 * than stubbed.
 */
export function CommandPalette({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const navigate = useNavigate();
  const [q, setQ] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setQ("");
    setCursor(0);
    const t = setTimeout(() => inputRef.current?.focus(), 0);
    return () => clearTimeout(t);
  }, [open]);

  const query = q.trim().toLowerCase();
  const pages = useMemo(
    () =>
      ALL_NAV_ITEMS.filter(
        (p) =>
          p.label.toLowerCase().includes(query) ||
          p.to.toLowerCase().includes(query)
      ).slice(0, 8),
    [query]
  );

  // Keep the cursor inside the (possibly shrunken) result list.
  const active = pages.length === 0 ? 0 : Math.min(cursor, pages.length - 1);

  if (!open) return null;

  const go = (to: string) => {
    navigate(to);
    onClose();
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") return onClose();
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setCursor((c) => (pages.length ? (c + 1) % pages.length : 0));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setCursor((c) =>
        pages.length ? (c - 1 + pages.length) % pages.length : 0
      );
    } else if (e.key === "Enter" && pages[active]) {
      e.preventDefault();
      go(pages[active].to);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[60] flex items-start justify-center bg-black/40 pt-[12vh] backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-label="Command palette"
        className="w-full max-w-xl overflow-hidden rounded-xl border border-border bg-popover shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 border-b border-border px-4">
          <Search className="h-4 w-4 shrink-0 text-muted-foreground" />
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              setCursor(0);
            }}
            onKeyDown={onKeyDown}
            placeholder="Search pages…"
            aria-label="Search pages"
            className="h-12 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
          <span className="rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
            esc
          </span>
        </div>

        <div className="max-h-[52vh] overflow-auto p-2">
          {pages.length === 0 ? (
            <div className="px-3 py-10 text-center text-sm text-muted-foreground">
              No matches for “{q}”.
            </div>
          ) : (
            <div className="space-y-0.5">
              {pages.map((p, i) => (
                <button
                  key={p.to}
                  type="button"
                  onMouseEnter={() => setCursor(i)}
                  onClick={() => go(p.to)}
                  className={`group flex w-full items-center gap-3 rounded-md px-3 py-2 text-left transition-colors ${
                    i === active ? "bg-muted" : "hover:bg-muted"
                  }`}
                >
                  <p.icon className="h-4 w-4 shrink-0 text-muted-foreground" />
                  <span className="flex min-w-0 flex-1 items-baseline gap-2">
                    <span className="truncate text-sm font-medium">
                      {p.label}
                    </span>
                    <span className="truncate font-mono text-xs text-muted-foreground">
                      {p.to}
                    </span>
                  </span>
                  {i === active && (
                    <CornerDownLeft className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="flex items-center justify-between border-t border-border bg-muted/30 px-4 py-2 text-[11px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <Command className="h-3 w-3" /> pages
          </span>
          <span className="flex items-center gap-1">
            <CornerDownLeft className="h-3 w-3" /> to open
          </span>
        </div>
      </div>
    </div>
  );
}
