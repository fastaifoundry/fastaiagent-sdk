import { Link } from "react-router-dom";
import type { LucideIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface DirectoryStat {
  label: string;
  value: string;
  /** Optional text color class (e.g. "text-fa-success", "text-destructive"). */
  accent?: string;
}

interface Props {
  /** Destination when the whole card is clicked. */
  to: string;
  icon: LucideIcon;
  title: string;
  /** Optional badge rendered on the right of the title (e.g. error indicator). */
  badge?: React.ReactNode;
  /** Optional one-line chip shown under the title (e.g. "chain · 3 nodes"). */
  chip?: React.ReactNode;
  /** 2×2 grid of stats shown as the card's primary content. */
  stats: DirectoryStat[];
  /** Optional footer row (e.g. "Updated 5m ago" or a path). */
  footer?: React.ReactNode;
}

/**
 * Shared card layout used by /agents, /workflows, /kb directory pages.
 * A click-through card with:
 *   [icon title         badge]
 *   [chip                    ]
 *   [stat stat]
 *   [stat stat]
 *   [footer row              ]
 *
 * Extracted in 0.9.4 after directory-page duplication crossed 3 copies.
 */
export function DirectoryCard({
  to,
  icon: Icon,
  title,
  badge,
  chip,
  stats,
  footer,
}: Props) {
  return (
    <Link to={to}>
      <Card className="h-full transition-colors hover:border-primary">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center justify-between text-sm">
            <span className="inline-flex items-center gap-2 truncate">
              <Icon className="h-4 w-4 shrink-0 text-muted-foreground" />
              {title}
            </span>
            {badge}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 pt-0">
          {chip}
          <dl className="grid grid-cols-2 gap-2 text-xs">
            {stats.map((s) => (
              <DirectoryStatView key={s.label} stat={s} />
            ))}
          </dl>
          {footer}
        </CardContent>
      </Card>
    </Link>
  );
}

function DirectoryStatView({ stat }: { stat: DirectoryStat }) {
  return (
    <div>
      <dt className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {stat.label}
      </dt>
      <dd className={cn("font-mono text-sm tabular-nums", stat.accent)}>
        {stat.value}
      </dd>
    </div>
  );
}
