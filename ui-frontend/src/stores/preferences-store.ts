import { create } from "zustand";
import { persist } from "zustand/middleware";

/**
 * Per-browser UI preferences that outlive a reload.
 *
 * ``includeExperiments`` (1.89.0): production views — Overview, Analytics,
 * cost breakdown, Agents — leave out traces whose ``fastaiagent.source`` is
 * a replay, Playground call, learn pass, eval run and so on, and report them
 * apart. Turning this on counts them as production again. Off by default so
 * a rerun is never mistaken for real traffic or real spend.
 */
interface PreferencesState {
  includeExperiments: boolean;
  setIncludeExperiments: (value: boolean) => void;
}

export const usePreferencesStore = create<PreferencesState>()(
  persist(
    (set) => ({
      includeExperiments: false,
      setIncludeExperiments: (value) => set({ includeExperiments: value }),
    }),
    {
      name: "fastaiagent-preferences",
      partialize: (s) => ({ includeExperiments: s.includeExperiments }),
    }
  )
);

/** The ``include_experiments`` preference, for hooks and pages. */
export function useIncludeExperiments(): boolean {
  return usePreferencesStore((s) => s.includeExperiments);
}
