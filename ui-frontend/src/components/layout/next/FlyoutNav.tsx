import { useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { Command, Home } from "lucide-react";
import { cn } from "@/lib/utils";
import { FastAIAgentLogo } from "@/components/brand/FastAIAgentLogo";
import { HOME, NAV_GROUPS, locate, type NavGroup } from "./nav-config";

/**
 * FlyoutNav — a slim icon rail; hovering (or clicking) a pillar pops a flyout
 * of its sections and items, which disappears on leave. No persistent panel,
 * so the main content gets the full width.
 *
 * Ported from the Enterprise /next shell. Two deliberate differences:
 *
 *  - Flyout items are <NavLink>s, not buttons. Enterprise uses
 *    <button onClick={navigate}>, which silently loses middle-click and
 *    open-in-new-tab, and makes the items unreachable via getByRole("link").
 *  - No /next path-prefix machinery — both Local UI skins serve the same
 *    routes, so there is nothing to rewrite.
 */
export function FlyoutNav({ onOpenPalette }: { onOpenPalette: () => void }) {
  const location = useLocation();
  const navigate = useNavigate();
  const activePillar = locate(location.pathname).group?.key;

  const [openKey, setOpenKey] = useState<string | null>(null);
  const [openTop, setOpenTop] = useState(0);
  const openGroup = NAV_GROUPS.find((g) => g.key === openKey);

  // Anchor the flyout to the hovered pillar, but never let it start so low
  // that a tall menu spills off-screen. The max-height is then bound to the
  // space actually left below, so the list scrolls rather than clipping.
  const FLYOUT_MARGIN = 8;
  const viewportH = typeof window === "undefined" ? 768 : window.innerHeight;
  const flyoutTop = Math.max(
    FLYOUT_MARGIN,
    Math.min(openTop, viewportH - 360)
  );
  const flyoutMaxH = viewportH - flyoutTop - FLYOUT_MARGIN;

  const open = (key: string, el: HTMLElement) => {
    setOpenKey(key);
    setOpenTop(el.getBoundingClientRect().top);
  };

  return (
    <aside
      className="relative flex h-full shrink-0"
      onMouseLeave={() => setOpenKey(null)}
    >
      <div className="flex h-full w-20 flex-col items-center border-r border-border bg-card py-3">
        <button
          type="button"
          onClick={() => navigate("/")}
          title="FastAIAgent"
          className="mb-2 shrink-0"
        >
          <FastAIAgentLogo variant="favicon" className="h-9 w-9" />
          <span className="sr-only">FastAIAgent</span>
        </button>

        <RailBtn
          active={location.pathname === "/"}
          label={HOME.label}
          onClick={() => navigate("/")}
        >
          <Home className="h-5 w-5" />
        </RailBtn>

        <div className="my-1 h-px w-6 shrink-0 bg-border" />

        {/* Scrollable pillar list — keeps Home pinned above, Search below. */}
        <div className="flex w-full min-h-0 flex-1 flex-col items-center gap-1 overflow-y-auto py-1 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {NAV_GROUPS.map((group) => (
            <div
              key={group.key}
              onMouseEnter={(e) => open(group.key, e.currentTarget)}
            >
              <RailBtn
                active={openKey === group.key}
                highlight={activePillar === group.key}
                label={group.railLabel ?? group.label}
                onClick={(e) =>
                  openKey === group.key
                    ? setOpenKey(null)
                    : open(group.key, e.currentTarget)
                }
              >
                <group.icon className="h-5 w-5" />
              </RailBtn>
            </div>
          ))}
        </div>

        <div className="shrink-0 pt-2">
          <RailBtn active={false} label="Search" onClick={onOpenPalette}>
            <Command className="h-5 w-5" />
          </RailBtn>
        </div>
      </div>

      {/* Fixed-positioned so the scrollable rail never clips it. */}
      {openGroup && (
        <Flyout
          group={openGroup}
          top={flyoutTop}
          maxHeight={flyoutMaxH}
          onNavigate={() => setOpenKey(null)}
        />
      )}
    </aside>
  );
}

function Flyout({
  group,
  top,
  maxHeight,
  onNavigate,
}: {
  group: NavGroup;
  top: number;
  maxHeight: number;
  onNavigate: () => void;
}) {
  return (
    <div
      className="fixed z-50 flex w-64 flex-col overflow-hidden rounded-lg border border-border bg-popover shadow-xl"
      style={{ top, left: 80, maxHeight }}
    >
      <div className="shrink-0 border-b border-border px-3 py-2.5">
        <div className="flex items-center gap-2">
          <group.icon className="h-4 w-4 shrink-0 text-primary" />
          <span className="text-sm font-semibold">{group.label}</span>
        </div>
        <p className="mt-0.5 text-xs text-muted-foreground">{group.blurb}</p>
      </div>
      <div className="min-h-0 flex-1 space-y-2 overflow-auto p-2">
        {group.sections.map((section, i) => (
          <div key={section.label ?? i} className="space-y-0.5">
            {section.label && (
              <div className="px-2.5 pb-0.5 pt-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                {section.label}
              </div>
            )}
            {section.items.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                onClick={onNavigate}
                className={({ isActive }) =>
                  cn(
                    "group flex w-full min-w-0 items-center gap-3 rounded-md px-2.5 py-2 text-sm transition-colors",
                    isActive
                      ? "bg-primary/10 text-primary font-medium"
                      : "text-foreground/80 hover:bg-muted"
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <item.icon
                      className={cn(
                        "h-4 w-4 shrink-0",
                        isActive ? "text-primary" : "text-muted-foreground"
                      )}
                    />
                    <span className="min-w-0 flex-1 truncate text-left leading-snug">
                      {item.label}
                    </span>
                  </>
                )}
              </NavLink>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function RailBtn({
  active,
  highlight,
  label,
  onClick,
  children,
}: {
  active?: boolean;
  highlight?: boolean;
  label: string;
  onClick: (e: React.MouseEvent<HTMLButtonElement>) => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      className={cn(
        "relative flex h-11 w-11 shrink-0 flex-col items-center justify-center rounded-xl text-[9px] font-medium transition-colors",
        active
          ? "bg-primary/10 text-primary"
          : "text-muted-foreground hover:bg-muted hover:text-foreground"
      )}
    >
      {active && (
        <span className="absolute left-0 top-1/2 h-6 w-0.5 -translate-y-1/2 rounded-r bg-primary" />
      )}
      {!active && highlight && (
        <span className="absolute right-1 top-1.5 h-1.5 w-1.5 rounded-full bg-primary" />
      )}
      {children}
      <span className="mt-0.5 leading-none">{label}</span>
    </button>
  );
}
