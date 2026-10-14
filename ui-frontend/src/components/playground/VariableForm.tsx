import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

interface Props {
  variables: string[];
  values: Record<string, string>;
  onChange: (next: Record<string, string>) => void;
  /** Variables a run would send unfilled, highlighted (1.89.0, P10e). */
  unfilled?: string[];
}

export function VariableForm({ variables, values, onChange, unfilled = [] }: Props) {
  if (variables.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        No variables detected. Add <code>{`{{name}}`}</code> placeholders to
        the template above.
      </p>
    );
  }
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      {variables.map((name) => {
        const empty = unfilled.includes(name);
        return (
          <div key={name} className="space-y-1">
            <Label htmlFor={`var-${name}`} className="font-mono text-xs">
              {`{{${name}}}`}
            </Label>
            <Input
              id={`var-${name}`}
              value={values[name] ?? ""}
              onChange={(e) =>
                onChange({ ...values, [name]: e.target.value })
              }
              placeholder={name}
              data-unfilled={empty || undefined}
              className={cn(empty && "border-amber-500/60")}
            />
          </div>
        );
      })}
    </div>
  );
}
