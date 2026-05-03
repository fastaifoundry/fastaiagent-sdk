import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import type { PlaygroundParameters } from "@/lib/types";

interface Props {
  value: PlaygroundParameters;
  onChange: (next: PlaygroundParameters) => void;
}

export function ParametersPanel({ value, onChange }: Props) {
  return (
    <div className="space-y-4 text-sm">
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <Label htmlFor="param-temperature" className="text-xs">
            temperature
          </Label>
          <span className="font-mono text-xs tabular-nums text-muted-foreground">
            {value.temperature.toFixed(2)}
          </span>
        </div>
        <Slider
          id="param-temperature"
          min={0}
          max={2}
          step={0.05}
          value={[value.temperature]}
          onValueChange={(v) =>
            onChange({ ...value, temperature: v[0] ?? 1 })
          }
        />
      </div>

      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <Label htmlFor="param-top-p" className="text-xs">
            top_p
          </Label>
          <span className="font-mono text-xs tabular-nums text-muted-foreground">
            {value.top_p.toFixed(2)}
          </span>
        </div>
        <Slider
          id="param-top-p"
          min={0}
          max={1}
          step={0.01}
          value={[value.top_p]}
          onValueChange={(v) => onChange({ ...value, top_p: v[0] ?? 1 })}
        />
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="param-max-tokens" className="text-xs">
          max_tokens
        </Label>
        <Input
          id="param-max-tokens"
          type="number"
          min={1}
          max={200000}
          value={value.max_tokens}
          onChange={(e) =>
            onChange({
              ...value,
              max_tokens: Number.parseInt(e.target.value, 10) || 1024,
            })
          }
        />
      </div>
    </div>
  );
}
