import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { Sidebar } from "./Sidebar";

describe("Sidebar", () => {
  it("renders every Local-tier nav entry", () => {
    renderWithProviders(<Sidebar />);
    expect(screen.getByRole("link", { name: /FastAIAgent/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Home/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Traces/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Guardrail Events/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Eval Runs/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Prompts/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Agents/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Knowledge Bases/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Workflows/i })).toBeInTheDocument();
  });

  it("does NOT render higher-tier Platform-only surfaces", () => {
    renderWithProviders(<Sidebar />);
    expect(screen.queryByRole("link", { name: /Chains/i })).toBeNull();
    expect(screen.queryByRole("link", { name: /Connectors/i })).toBeNull();
    expect(screen.queryByRole("link", { name: /Billing/i })).toBeNull();
    expect(screen.queryByRole("link", { name: /Admin/i })).toBeNull();
  });

  it("marks the active route", () => {
    renderWithProviders(<Sidebar />, { route: "/traces" });
    const activeLink = screen.getByRole("link", { name: /Traces/i });
    // NavLink adds an active class — we check for the token we use.
    expect(activeLink.className).toMatch(/sidebar-item-active|text-primary/);
  });
});
