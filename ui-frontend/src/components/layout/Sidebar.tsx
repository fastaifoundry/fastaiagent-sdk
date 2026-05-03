import { Link, NavLink } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Bot,
  CheckSquare,
  Database,
  FileText,
  GitBranch,
  LayoutDashboard,
  Play,
  Shield,
  TrendingUp,
} from "lucide-react";
import { FastAIAgentLogo } from "@/components/brand/FastAIAgentLogo";

interface NavItem {
  to: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  end?: boolean;
}

interface NavSection {
  label: string;
  items: NavItem[];
}

const SECTIONS: NavSection[] = [
  {
    label: "// OVERVIEW",
    items: [{ to: "/", label: "Home", icon: LayoutDashboard, end: true }],
  },
  {
    label: "// OBSERVABILITY",
    items: [
      { to: "/traces", label: "Traces", icon: Activity },
      { to: "/analytics", label: "Analytics", icon: BarChart3 },
      { to: "/guardrails", label: "Guardrail Events", icon: Shield },
    ],
  },
  {
    label: "// HITL",
    items: [{ to: "/approvals", label: "Approvals", icon: CheckSquare }],
  },
  {
    label: "// EVALUATION",
    items: [{ to: "/evals", label: "Eval Runs", icon: TrendingUp }],
  },
  {
    label: "// PROMPT REGISTRY",
    items: [
      { to: "/prompts", label: "Prompts", icon: FileText },
      { to: "/playground", label: "Playground", icon: Play },
    ],
  },
  {
    label: "// KNOWLEDGE",
    items: [{ to: "/kb", label: "Knowledge Bases", icon: Database }],
  },
  {
    label: "// WORKFLOWS & AGENTS",
    items: [
      { to: "/workflows", label: "Workflows", icon: GitBranch },
      { to: "/agents", label: "Agents", icon: Bot },
    ],
  },
];

export function Sidebar() {
  return (
    <aside className="flex h-screen w-60 flex-col border-r border-sidebar-border bg-card">
      <Link
        to="/"
        className="flex h-14 items-center gap-2 border-b border-sidebar-border px-4 hover:bg-muted/50 transition-colors"
      >
        <FastAIAgentLogo className="h-9 w-9 rounded" variant="favicon" />
        <span className="text-lg font-semibold tracking-tight">FastAIAgent</span>
      </Link>

      <nav className="flex-1 overflow-auto px-3 py-4 space-y-5">
        {SECTIONS.map((section) => (
          <div key={section.label}>
            <div className="mb-2 px-3 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              {section.label}
            </div>
            <div className="space-y-0.5">
              {section.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) =>
                    `group flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                      isActive
                        ? "bg-primary/10 text-primary border-l-2 border-primary sidebar-item-active"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground border-l-2 border-transparent"
                    }`
                  }
                >
                  <item.icon className="h-4 w-4 shrink-0" />
                  {item.label}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>
    </aside>
  );
}
