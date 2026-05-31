import { Link } from "react-router-dom";
import { ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";
import type { TranscriptTurn } from "@/lib/types";

/**
 * Linear chat-bubble view of a simulated conversation. Plain Tailwind divs —
 * the transcript is always linear (user ↔ assistant), so no graph library is
 * needed. Each assistant turn deep-links to its trace.
 */
export function SimulationTranscriptView({
  transcript,
}: {
  transcript: TranscriptTurn[];
}) {
  if (!transcript.length) {
    return (
      <p className="px-2 py-4 text-xs text-muted-foreground">
        No transcript recorded.
      </p>
    );
  }

  return (
    <div className="space-y-2">
      {transcript.map((turn) => {
        const isUser = turn.role === "user";
        return (
          <div
            key={turn.turn_index}
            className={cn("flex", isUser ? "justify-start" : "justify-end")}
          >
            <div
              className={cn(
                "max-w-[80%] rounded-lg px-3 py-2 text-sm",
                isUser
                  ? "bg-muted text-foreground"
                  : "bg-primary/10 text-foreground"
              )}
            >
              <div className="mb-1 flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
                <span>{isUser ? "user" : "assistant"}</span>
                {turn.trace_id && (
                  <Link
                    to={`/traces/${turn.trace_id}`}
                    title="View trace"
                    className="inline-flex items-center gap-0.5 normal-case text-muted-foreground hover:text-primary"
                  >
                    view trace
                    <ExternalLink className="h-3 w-3" />
                  </Link>
                )}
              </div>
              <div className="whitespace-pre-wrap break-words">
                {turn.content}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
