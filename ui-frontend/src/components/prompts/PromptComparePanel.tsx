import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TableSkeleton } from "@/components/shared/LoadingSkeleton";
import { usePromptDiff } from "@/hooks/use-prompts";
import { cn } from "@/lib/utils";

interface Props {
  slug: string;
  /** Every stored version, ascending. They need not be contiguous. */
  versions: number[];
  /** The version open in the editor: the default right-hand side. */
  selected: number | null;
}

function lineClass(line: string): string {
  if (line.startsWith("+++") || line.startsWith("---")) return "text-muted-foreground font-semibold";
  if (line.startsWith("@@")) return "text-fa-info";
  if (line.startsWith("+")) return "bg-fa-success/10 text-fa-success";
  if (line.startsWith("-")) return "bg-destructive/10 text-destructive";
  return "";
}

/**
 * Compare two versions of a prompt (1.89.0, P10b).
 *
 * The server's diff is a unified diff over lines (it used to print both whole
 * templates as one -/+ pair). Below it, both stored templates side by side,
 * `{{@fragment}}` references intact.
 */
export function PromptComparePanel({ slug, versions, selected }: Props) {
  const [open, setOpen] = useState(false);
  const [a, setA] = useState<number | null>(null);
  const [b, setB] = useState<number | null>(null);

  // Default: the version before the selected one, against the selected one.
  const right = b ?? selected ?? versions.at(-1) ?? null;
  const before = versions.filter((v) => right != null && v < right);
  const left = a ?? before.at(-1) ?? versions.find((v) => v !== right) ?? null;

  const diff = usePromptDiff(open ? slug : undefined, left, right);

  const picker = (id: string, label: string, value: number | null, set: (v: number) => void) => (
    <label className="flex items-center gap-1.5 text-xs" htmlFor={id}>
      {label}
      <select
        id={id}
        value={value ?? ""}
        onChange={(e) => set(Number(e.target.value))}
        className="h-8 rounded-md border border-input bg-background px-2 font-mono text-xs"
      >
        {versions.map((v) => (
          <option key={v} value={v}>
            v{v}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">
          <button
            type="button"
            className="flex items-center gap-1"
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            Compare versions
          </button>
        </CardTitle>
      </CardHeader>
      {open && (
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            {picker("compare-a", "From", left, setA)}
            {picker("compare-b", "To", right, setB)}
          </div>
          {left === right ? (
            <p className="text-xs text-muted-foreground">Pick two different versions.</p>
          ) : diff.isLoading ? (
            <TableSkeleton rows={3} />
          ) : diff.isError ? (
            <p className="text-xs text-destructive">
              {diff.error instanceof Error ? diff.error.message : "Could not load the diff"}
            </p>
          ) : diff.data ? (
            <>
              <pre
                className="max-h-80 overflow-auto rounded-md border bg-muted/20 p-2 font-mono text-xs"
                data-testid="prompt-diff"
              >
                {diff.data.diff.split("\n").map((line, i) => (
                  <div key={i} className={cn("whitespace-pre-wrap px-1", lineClass(line))}>
                    {line || " "}
                  </div>
                ))}
              </pre>
              <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                {[diff.data.a, diff.data.b].map((side) => (
                  <div key={side.version} className="min-w-0 space-y-1">
                    <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                      v{side.version}
                    </div>
                    <pre
                      className="max-h-64 overflow-auto rounded-md border p-2 font-mono text-xs whitespace-pre-wrap"
                      data-testid={`prompt-side-${side.version}`}
                    >
                      {side.template}
                    </pre>
                  </div>
                ))}
              </div>
            </>
          ) : null}
        </CardContent>
      )}
    </Card>
  );
}
