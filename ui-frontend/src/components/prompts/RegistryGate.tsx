import { Info } from "lucide-react";

export function RegistryExternalBanner() {
  return (
    <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-sm text-amber-900 dark:text-amber-200">
      <Info className="mt-0.5 h-4 w-4 shrink-0" />
      <div>
        <p className="font-medium">This registry is external.</p>
        <p className="text-xs text-amber-900/80 dark:text-amber-200/70">
          Edit prompts via code or from the environment that owns this path.
        </p>
      </div>
    </div>
  );
}
