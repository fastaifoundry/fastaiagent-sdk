import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { isValidMaxTokens, MAX_TOKENS_MAX, MAX_TOKENS_MIN } from "@/lib/playground-params";
import { FREE_RULE } from "@/lib/sampling-rules";
import type { ParamSupport, PlaygroundParameters, SamplingRule } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  value: PlaygroundParameters;
  onChange: (next: PlaygroundParameters) => void;
  /**
   * What the selected model accepts (1.89.0, P7), resolved from the server's
   * table. Hints only: the server forwards whatever the run sends.
   */
  rule?: SamplingRule;
}

interface OptionalSliderProps {
  id: string;
  label: string;
  value: number | null;
  min: number;
  max: number;
  step: number;
  /** Shown while off, and what switching on sends: the model's default. */
  fallback: number;
  support: ParamSupport;
  note: string;
  onChange: (next: number | null) => void;
}

/**
 * A slider that can be left unset.
 *
 * Off means the parameter isn't sent and the provider applies its own
 * default. `unsupported` keeps it off (the model rejects any value);
 * `default_only` lets it be sent, but only at the default.
 */
function OptionalSlider({
  id,
  label,
  value,
  min,
  max,
  step,
  fallback,
  support,
  note,
  onChange,
}: OptionalSliderProps) {
  const unsupported = support === "unsupported";
  const pinned = support === "default_only";
  const enabled = value !== null && !unsupported;
  const shown = enabled ? (pinned ? fallback : value) : fallback;
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2" title={unsupported || pinned ? note : undefined}>
          <Switch
            id={`${id}-enabled`}
            checked={enabled}
            disabled={unsupported}
            onCheckedChange={(on) => onChange(on ? fallback : null)}
            aria-label={`Send ${label}`}
          />
          <Label htmlFor={id} className="text-xs">
            {label}
          </Label>
        </div>
        <span className="font-mono text-xs tabular-nums text-muted-foreground">
          {unsupported ? "not sent" : enabled ? shown.toFixed(2) : "auto"}
        </span>
      </div>
      {/* Dimmed when off so a filled track doesn't read as "a value is set"
          while the readout says "auto". */}
      <Slider
        id={id}
        min={min}
        max={max}
        step={step}
        disabled={!enabled || pinned}
        className={cn((!enabled || pinned) && "opacity-40")}
        value={[shown]}
        onValueChange={(v) => onChange(v[0] ?? fallback)}
      />
      {(unsupported || pinned) && (
        <p className="text-[11px] text-muted-foreground" data-testid={`${id}-rule`}>
          {unsupported
            ? `This model does not accept ${label}.`
            : `This model accepts only the default ${label} (${fallback}).`}
        </p>
      )}
    </div>
  );
}

export function ParametersPanel({ value, onChange, rule = FREE_RULE }: Props) {
  const [rawMaxTokens, setRawMaxTokens] = useState<string>(String(value.max_tokens));
  // Follow external changes (a history entry restored, a reset) without
  // clobbering what the user is typing.
  useEffect(() => {
    if (Number(rawMaxTokens) !== value.max_tokens) {
      setRawMaxTokens(String(value.max_tokens));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value.max_tokens]);
  const maxTokensValid = rawMaxTokens.trim() !== "" && isValidMaxTokens(Number(rawMaxTokens));
  const bothOn = value.temperature !== null && value.top_p !== null;
  return (
    <div className="space-y-4 text-sm">
      <p className="text-xs text-muted-foreground">
        Off means the parameter isn&apos;t sent, so the provider&apos;s own default applies.
        Limits known for the selected model are shown here; the provider has the final say.
      </p>
      {rule.note && (
        <p
          className="rounded-md border border-amber-500/30 bg-amber-500/5 px-2 py-1.5 text-[11px] text-amber-900 dark:text-amber-200"
          data-testid="sampling-rule-note"
        >
          {rule.note}
        </p>
      )}

      <OptionalSlider
        id="param-temperature"
        label="temperature"
        value={value.temperature}
        min={0}
        max={2}
        step={0.05}
        fallback={rule.default_temperature}
        support={rule.temperature}
        note={rule.note}
        // Under an exclusive rule, switching one on switches the other off.
        onChange={(temperature) =>
          onChange({
            ...value,
            temperature,
            ...(rule.exclusive && temperature !== null ? { top_p: null } : {}),
          })
        }
      />

      <OptionalSlider
        id="param-top-p"
        label="top_p"
        value={value.top_p}
        min={0}
        max={1}
        step={0.01}
        fallback={rule.default_top_p}
        support={rule.top_p}
        note={rule.note}
        onChange={(top_p) =>
          onChange({
            ...value,
            top_p,
            ...(rule.exclusive && top_p !== null ? { temperature: null } : {}),
          })
        }
      />

      {rule.exclusive && bothOn && (
        <p className="text-[11px] text-amber-700 dark:text-amber-300" role="alert">
          This model accepts temperature or top_p, not both. Switch one off.
        </p>
      )}

      <div className="space-y-1.5">
        <Label htmlFor="param-max-tokens" className="text-xs">
          max_tokens
        </Label>
        <Input
          id="param-max-tokens"
          type="number"
          min={MAX_TOKENS_MIN}
          max={MAX_TOKENS_MAX}
          value={rawMaxTokens}
          aria-invalid={!maxTokensValid}
          onChange={(e) => {
            // Keep what was typed. Snapping a bad value to 1024 hid the
            // mistake; sending it produced a 422 (1.89.0, P1).
            const raw = e.target.value;
            setRawMaxTokens(raw);
            onChange({ ...value, max_tokens: Number(raw) });
          }}
        />
        {!maxTokensValid && (
          <p className="text-[11px] text-destructive" role="alert">
            max_tokens must be a whole number from {MAX_TOKENS_MIN} to{" "}
            {MAX_TOKENS_MAX.toLocaleString()}.
          </p>
        )}
      </div>
    </div>
  );
}
