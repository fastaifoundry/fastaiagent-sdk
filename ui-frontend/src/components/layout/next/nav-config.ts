import {
  Activity,
  BarChart3,
  Bot,
  Brain,
  CheckSquare,
  ClipboardCheck,
  Database,
  Eye,
  FileText,
  GitBranch,
  Hammer,
  LayoutDashboard,
  ListChecks,
  MessagesSquare,
  Play,
  Scale,
  Shield,
  Sparkles,
  TrendingUp,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  end?: boolean;
}

export interface NavSection {
  label?: string;
  items: NavItem[];
}

export interface NavGroup {
  key: string;
  label: string;
  /** Shorter label for the icon rail, where space is tight. */
  railLabel?: string;
  icon: LucideIcon;
  blurb: string;
  sections: NavSection[];
}

/**
 * The four lifecycle pillars, plus a pinned Home.
 *
 * These are the same 14 destinations the classic Sidebar lists — regrouped
 * from 7 flat sections into 4 pillars that follow the agent lifecycle
 * (build it → test it → watch it → govern it), matching the Enterprise
 * console's information architecture.
 *
 * Enterprise also has Compliance and Settings pillars; the Local UI has no
 * routes behind either, and an empty pillar is worse than an absent one.
 *
 * Every entry here must stay reachable — FlyoutNav.test.tsx asserts this list
 * covers every link the classic Sidebar offers.
 */
export const NAV_GROUPS: NavGroup[] = [
  {
    key: "build",
    label: "Build",
    icon: Hammer,
    blurb: "Agents, prompts, knowledge & workflows",
    sections: [
      {
        label: "Fleet",
        items: [
          { to: "/agents", label: "Agents", icon: Bot },
          { to: "/workflows", label: "Workflows", icon: GitBranch },
        ],
      },
      {
        label: "Prompts",
        items: [
          { to: "/prompts", label: "Prompts", icon: FileText },
          { to: "/playground", label: "Playground", icon: Play },
        ],
      },
      {
        label: "Knowledge",
        items: [
          { to: "/kb", label: "Knowledge Bases", icon: Database },
          { to: "/memory", label: "Memory", icon: Brain },
        ],
      },
    ],
  },
  {
    key: "evaluate",
    label: "Evaluate",
    icon: ClipboardCheck,
    blurb: "Test, simulate & optimise",
    sections: [
      {
        items: [
          { to: "/evals", label: "Eval Runs", icon: TrendingUp },
          { to: "/simulations", label: "Simulations", icon: MessagesSquare },
          { to: "/optimizes", label: "AutoLLM", icon: Sparkles },
          { to: "/datasets", label: "Datasets", icon: ListChecks },
        ],
      },
    ],
  },
  {
    key: "observe",
    label: "Observe",
    icon: Eye,
    blurb: "Traces & analytics",
    sections: [
      {
        items: [
          { to: "/traces", label: "Traces", icon: Activity },
          { to: "/analytics", label: "Analytics", icon: BarChart3 },
        ],
      },
    ],
  },
  {
    key: "govern",
    label: "Govern",
    icon: Scale,
    blurb: "Guardrails & human review",
    sections: [
      {
        items: [
          { to: "/guardrails", label: "Guardrail Events", icon: Shield },
          { to: "/approvals", label: "Approvals", icon: CheckSquare },
        ],
      },
    ],
  },
];

export const HOME: NavItem = {
  to: "/",
  label: "Home",
  icon: LayoutDashboard,
  end: true,
};

/** Home plus every pillar item, flattened — used by the command palette. */
export const ALL_NAV_ITEMS: NavItem[] = [
  HOME,
  ...NAV_GROUPS.flatMap((g) => g.sections.flatMap((s) => s.items)),
];

/**
 * Resolve a pathname to its pillar / section / item, for rail highlighting and
 * the header breadcrumb. Matches the longest `to` prefix so detail routes
 * (e.g. /traces/abc123) resolve to their list page.
 */
export function locate(pathname: string): {
  group?: NavGroup;
  section?: string;
  item?: NavItem;
} {
  if (pathname === "/") return { item: HOME };

  let best: { group: NavGroup; section?: string; item: NavItem } | undefined;
  for (const group of NAV_GROUPS) {
    for (const section of group.sections) {
      for (const item of section.items) {
        if (pathname === item.to || pathname.startsWith(item.to + "/")) {
          if (!best || item.to.length > best.item.to.length) {
            best = { group, section: section.label, item };
          }
        }
      }
    }
  }
  return best ?? {};
}
