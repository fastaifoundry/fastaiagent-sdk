import { useEffect, useState } from "react";
import { Loader2, StickyNote } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Timestamp } from "@/components/shared/Timestamp";
import { useDeleteTraceNote, useSaveTraceNote } from "@/hooks/use-traces";
import { ApiError } from "@/lib/api";
import type { TraceNote } from "@/lib/types";

/** The server's cap (`POST /api/traces/{id}/notes`, max_length=20000). */
export const NOTE_MAX_CHARS = 20_000;

interface Props {
  traceId: string;
  note: TraceNote | null | undefined;
}

/**
 * A free-text note on a trace (1.89.0).
 *
 * The notes API had no UI. Noted traces are what
 * `curate_from_traces(filter="noted")` turns into eval cases, so a note is
 * how you flag a trace worth keeping. Saving an empty note deletes it.
 */
export function TraceNotesCard({ traceId, note }: Props) {
  const saved = note?.text ?? "";
  const [draft, setDraft] = useState(saved);
  const save = useSaveTraceNote(traceId);
  const clear = useDeleteTraceNote(traceId);

  // Follow the stored note when it changes (saved here, or a refresh).
  useEffect(() => {
    setDraft(saved);
  }, [saved]);

  const dirty = draft !== saved;
  const busy = save.isPending || clear.isPending;

  const handleSave = async () => {
    try {
      const res = await save.mutateAsync(draft);
      toast.success(res.note ? "Note saved" : "Note removed");
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : "Could not save the note");
    }
  };

  const handleClear = async () => {
    try {
      await clear.mutateAsync();
      setDraft("");
      toast.success("Note removed");
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : "Could not remove the note");
    }
  };

  return (
    <Card data-testid="trace-notes-card">
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-sm">
          <span className="flex items-center gap-1.5">
            <StickyNote className="h-3.5 w-3.5" />
            Note
          </span>
          {note?.updated_at && (
            <span className="text-xs font-normal text-muted-foreground">
              updated <Timestamp iso={note.updated_at} />
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <Textarea
          aria-label="Trace note"
          value={draft}
          onChange={(e) => setDraft(e.target.value.slice(0, NOTE_MAX_CHARS))}
          maxLength={NOTE_MAX_CHARS}
          rows={3}
          placeholder="Why this trace matters: a bug, a good answer, a case to keep…"
          className="text-sm"
        />
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs text-muted-foreground">
            Noted traces feed{" "}
            <code className="font-mono">curate_from_traces(filter=&quot;noted&quot;)</code>.{" "}
            <span className="tabular-nums" data-testid="note-counter">
              {draft.length.toLocaleString()} / {NOTE_MAX_CHARS.toLocaleString()}
            </span>
          </p>
          <div className="flex gap-2">
            {note && (
              <Button variant="ghost" size="sm" onClick={handleClear} disabled={busy}>
                Clear
              </Button>
            )}
            <Button size="sm" onClick={handleSave} disabled={!dirty || busy}>
              {save.isPending && <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />}
              Save note
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
