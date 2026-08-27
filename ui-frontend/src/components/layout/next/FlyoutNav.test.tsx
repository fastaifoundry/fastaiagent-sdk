import { describe, expect, it } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { FlyoutNav } from "./FlyoutNav";
import { ALL_NAV_ITEMS, NAV_GROUPS, locate } from "./nav-config";

/**
 * Mirrors Sidebar.test.tsx for the new shell, plus the reachability invariant:
 * regrouping 7 flat sections into 4 flyout pillars must not drop a destination.
 */

/** Every top-level link the classic Sidebar offers. */
const CLASSIC_SIDEBAR_ROUTES = [
  "/",
  "/traces",
  "/analytics",
  "/guardrails",
  "/approvals",
  "/evals",
  "/simulations",
  "/optimizes",
  "/datasets",
  "/prompts",
  "/playground",
  "/kb",
  "/memory",
  "/workflows",
  "/agents",
];

function openPillar(label: string) {
  fireEvent.click(screen.getByTitle(label));
}

describe("nav-config", () => {
  it("covers every route the classic sidebar links to", () => {
    const reachable = new Set(ALL_NAV_ITEMS.map((i) => i.to));
    const missing = CLASSIC_SIDEBAR_ROUTES.filter((r) => !reachable.has(r));
    expect(missing).toEqual([]);
  });

  it("does NOT introduce higher-tier Platform-only surfaces", () => {
    const labels = ALL_NAV_ITEMS.map((i) => i.label);
    expect(labels).not.toContain("Chains");
    expect(labels).not.toContain("Connectors");
    expect(labels).not.toContain("Billing");
    expect(labels).not.toContain("Admin");
  });

  it("lists no route twice", () => {
    const routes = ALL_NAV_ITEMS.map((i) => i.to);
    expect(routes.length).toBe(new Set(routes).size);
  });

  it("resolves detail routes to their list page's pillar", () => {
    expect(locate("/traces/abc123").item?.to).toBe("/traces");
    expect(locate("/traces/abc123").group?.key).toBe("observe");
    expect(locate("/").item?.label).toBe("Home");
  });
});

describe("FlyoutNav", () => {
  it("renders a rail button for Home, every pillar, and search", () => {
    renderWithProviders(<FlyoutNav onOpenPalette={() => {}} />);
    expect(screen.getByTitle("Home")).toBeInTheDocument();
    for (const group of NAV_GROUPS) {
      expect(screen.getByTitle(group.railLabel ?? group.label)).toBeInTheDocument();
    }
    expect(screen.getByTitle("Search")).toBeInTheDocument();
  });

  it("reveals a pillar's links as real anchors when opened", () => {
    renderWithProviders(<FlyoutNav onOpenPalette={() => {}} />);

    // Closed by default — the flyout only mounts on hover/click.
    expect(screen.queryByRole("link", { name: /Playground/i })).toBeNull();

    openPillar("Build");
    for (const label of [
      "Agents",
      "Workflows",
      "Prompts",
      "Playground",
      "Knowledge Bases",
      "Memory",
    ]) {
      // Anchors, not buttons — preserves middle-click and open-in-new-tab.
      expect(screen.getByRole("link", { name: label })).toHaveAttribute("href");
    }
  });

  it("exposes every nav destination across its pillars", () => {
    renderWithProviders(<FlyoutNav onOpenPalette={() => {}} />);
    for (const group of NAV_GROUPS) {
      openPillar(group.railLabel ?? group.label);
      for (const item of group.sections.flatMap((s) => s.items)) {
        expect(screen.getByRole("link", { name: item.label })).toBeInTheDocument();
      }
    }
  });

  it("marks the active route inside the flyout", () => {
    renderWithProviders(<FlyoutNav onOpenPalette={() => {}} />, {
      route: "/traces",
    });
    openPillar("Observe");
    const active = screen.getByRole("link", { name: "Traces" });
    expect(active.className).toMatch(/text-primary/);
  });

  it("calls onOpenPalette when search is clicked", () => {
    let opened = false;
    renderWithProviders(<FlyoutNav onOpenPalette={() => (opened = true)} />);
    fireEvent.click(screen.getByTitle("Search"));
    expect(opened).toBe(true);
  });
});
