import { useState } from "react";
import { BookmarkPlus, Settings2, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  useCreateFilterPreset,
  useDeleteFilterPreset,
  useFilterPresets,
} from "@/hooks/use-filter-presets";
import type { FilterPreset, TraceFilters } from "@/lib/types";

interface Props {
  filters: TraceFilters;
  onApply: (filters: TraceFilters) => void;
}

/**
 * Saved-preset dropdown + "Save preset" + "Manage" controls for the
 * trace filter bar. The dropdown shows the project's presets; picking
 * one applies its filters via ``onApply`` (which also pushes them into
 * the URL via ``TracesPage``).
 *
 * Apply preserves ``page_size`` from the current filters because the
 * preset only stores filter dimensions, not pagination state.
 */
export function FilterPresetsMenu({ filters, onApply }: Props) {
  const presets = useFilterPresets();
  const create = useCreateFilterPreset();
  const del = useDeleteFilterPreset();

  const [saveOpen, setSaveOpen] = useState(false);
  const [manageOpen, setManageOpen] = useState(false);
  const [name, setName] = useState("");

  const list = presets.data ?? [];

  const handleApply = (p: FilterPreset) => {
    // Drop the preset's pagination so picking a preset starts on page 1
    // with the page_size the user is currently using.
    const { page: _p, page_size: _ps, ...rest } = p.filters;
    void _p;
    void _ps;
    onApply({
      ...rest,
      page: 1,
      page_size: filters.page_size ?? 100,
    });
  };

  const handleSave = async () => {
    if (!name.trim()) return;
    try {
      // Strip pagination — preset captures dimensions, not page state.
      const { page: _p, page_size: _ps, ...rest } = filters;
      void _p;
      void _ps;
      await create.mutateAsync({ name: name.trim(), filters: rest });
      toast.success(`Saved preset '${name.trim()}'`);
      setSaveOpen(false);
      setName("");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Save failed");
    }
  };

  return (
    <>
      <select
        value=""
        onChange={(e) => {
          const id = e.target.value;
          if (!id) return;
          const p = list.find((x) => x.id === id);
          if (p) handleApply(p);
          // Reset the visible value so the next "select" event fires
          // even when the user picks the same preset again.
          e.target.value = "";
        }}
        className="h-9 rounded-md border border-input bg-background px-2 text-sm"
        title={list.length === 0 ? "No saved presets" : "Load preset"}
        disabled={list.length === 0}
      >
        <option value="">{list.length === 0 ? "No presets" : "Load preset…"}</option>
        {list.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
          </option>
        ))}
      </select>

      <Button
        variant="outline"
        size="sm"
        onClick={() => setSaveOpen(true)}
      >
        <BookmarkPlus className="mr-1 h-3.5 w-3.5" />
        Save preset
      </Button>

      {list.length > 0 && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setManageOpen(true)}
          title="Manage presets"
        >
          <Settings2 className="h-3.5 w-3.5" />
        </Button>
      )}

      <Dialog open={saveOpen} onOpenChange={setSaveOpen}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>Save filter preset</DialogTitle>
            <DialogDescription>
              Captures every active filter (search, time range, status,
              duration, cost…) so you can re-apply them in one click.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Label htmlFor="preset-name">Name</Label>
            <Input
              id="preset-name"
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Errors this week"
            />
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setSaveOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={!name.trim() || create.isPending}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={manageOpen} onOpenChange={setManageOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Manage presets</DialogTitle>
            <DialogDescription>
              Delete presets you no longer need. Renaming and editing
              filters is a follow-up — for now, save a new one with the
              updated filters and delete the old.
            </DialogDescription>
          </DialogHeader>
          <ul className="divide-y rounded-md border">
            {list.map((p) => (
              <li
                key={p.id}
                className="flex items-center justify-between gap-2 px-3 py-2"
              >
                <span className="truncate text-sm">{p.name}</span>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 text-muted-foreground hover:text-destructive"
                  onClick={async () => {
                    try {
                      await del.mutateAsync(p.id);
                      toast.success(`Deleted '${p.name}'`);
                    } catch (e) {
                      toast.error(
                        e instanceof Error ? e.message : "Delete failed"
                      );
                    }
                  }}
                  title="Delete"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </li>
            ))}
          </ul>
        </DialogContent>
      </Dialog>
    </>
  );
}
