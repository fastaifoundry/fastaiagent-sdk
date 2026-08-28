import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import type { PlaygroundParameters } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  value: PlaygroundParameters;
  onChange: (next: PlaygroundParameters) => void;
}

/** Shown on the slider when a parameter is off; also the value we send once
 *  it's switched on. Providers apply their own default when we omit it. */
const FALLBACK = { temperature: 1, top_p: 1 } as const;

interface OptionalSliderProps {
  id: string;
  label: string;
  value: number | null;
  min: number;
  max: number;
  step: number;
  fallback: number;
  onChange: (next: number | null) => void;
}

/**
 * A slider that can be left unset.
 *
 * `temperature` and `top_p` are off by default on purpose: Anthropic rejects
 * both together, and Claude 5 rejects `top_p` at all. Sending them only when
 * the user asks for them keeps every provider working out of the box.
 */
function OptionalSlider({
  id,
  label,
  value,
  min,
  max,
  step,
  fallback,
  onChange,
}: OptionalSliderProps) {
  const enabled = value !== null;
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Switch
            id={`${id}-enabled`}
            checked={enabled}
            onCheckedChange={(on) => onChange(on ? fallback : null)}
            aria-label={`Send ${label}`}
          />
          <Label htmlFor={id} className="text-xs">
            {label}
          </Label>
        </div>
        <span className="font-mono text-xs tabular-nums text-muted-foreground">
          {enabled ? value.toFixed(2) : "auto"}
        </span>
      </div>
      {/* Dimmed when off so a filled track doesn't read as "a value is set"
          while the readout says "auto". */}
      <Slider
        id={id}
        min={min}
        max={max}
        step={step}
        disabled={!enabled}
        className={cn(!enabled && "opacity-40")}
        value={[enabled ? value : fallback]}
        onValueChange={(v) => onChange(v[0] ?? fallback)}
      />
    </div>
  );
}

export function ParametersPanel({ value, onChange }: Props) {
  return (
    <div className="space-y-4 text-sm">
      <p className="text-xs text-muted-foreground">
        Off means the parameter isn&apos;t sent, so the provider&apos;s own
        default applies. Anthropic rejects temperature and top_p together.
      </p>

      <OptionalSlider
        id="param-temperature"
        label="temperature"
        value={value.temperature}
        min={0}
        max={2}
        step={0.05}
        fallback={FALLBACK.temperature}
        onChange={(temperature) => onChange({ ...value, temperature })}
      />

      <OptionalSlider
        id="param-top-p"
        label="top_p"
        value={value.top_p}
        min={0}
        max={1}
        step={0.01}
        fallback={FALLBACK.top_p}
        onChange={(top_p) => onChange({ ...value, top_p })}
      />

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
