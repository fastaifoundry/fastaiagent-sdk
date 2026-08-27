import { getSkin, setSkin, type Skin } from "@/lib/skin";

const OPTIONS: { value: Skin; label: string; title: string }[] = [
  {
    value: "next",
    label: "New",
    title: "New UI — icon rail, flyout nav, Trace Studio palette",
  },
  {
    value: "classic",
    label: "Classic",
    title: "Classic UI — the original sidebar and Clean Lab palette",
  },
];

/**
 * Segmented New / Classic toggle. Switching persists the choice and reloads
 * (see lib/skin.ts for why), so this needs no state of its own.
 */
export function SkinSwitcher() {
  const current = getSkin();

  return (
    <div
      className="inline-flex items-center rounded-md border border-border bg-muted/40 p-0.5"
      role="group"
      aria-label="UI skin"
      data-testid="skin-switcher"
    >
      {OPTIONS.map((option) => {
        const active = current === option.value;
        return (
          <button
            key={option.value}
            type="button"
            title={option.title}
            aria-pressed={active}
            onClick={() => !active && setSkin(option.value)}
            className={`rounded px-2 py-0.5 text-[11px] font-medium transition-colors ${
              active
                ? "bg-card text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
