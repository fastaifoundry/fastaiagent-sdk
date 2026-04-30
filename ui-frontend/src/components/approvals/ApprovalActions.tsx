import { useState } from "react";
import { Check, X } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { useResumeExecution } from "@/hooks/use-approvals";

interface ApprovalActionsProps {
  executionId: string;
  /** Disable the buttons (e.g. during the resume call). */
  disabled?: boolean;
}

/** Approve / Reject buttons + reason textarea.
 *
 * Approve calls ``POST /api/executions/:id/resume`` with
 * ``{"approved": true, "metadata": {"reason": <text>}}``. Reject sends
 * ``approved: false``. After either succeeds, navigates to
 * ``/executions/:id`` so the operator can verify the run completed.
 */
export function ApprovalActions({ executionId, disabled }: ApprovalActionsProps) {
  const [reason, setReason] = useState("");
  const [error, setError] = useState<string | null>(null);
  const resume = useResumeExecution(executionId);
  const navigate = useNavigate();

  const onClick = async (approved: boolean) => {
    setError(null);
    try {
      await resume.mutateAsync({
        approved,
        metadata: reason ? { reason } : {},
      });
      navigate(`/executions/${executionId}`);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setError(message);
    }
  };

  return (
    <div className="space-y-3">
      <label className="block text-xs font-mono uppercase tracking-widest text-muted-foreground">
        Reason (optional)
      </label>
      <textarea
        value={reason}
        onChange={(e) => setReason(e.target.value)}
        rows={3}
        disabled={disabled || resume.isPending}
        placeholder="Why are you approving / rejecting? Stored as Resume(metadata={reason: ...})."
        className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50"
      />
      <div className="flex gap-2">
        <Button
          onClick={() => onClick(true)}
          disabled={disabled || resume.isPending}
          className="bg-green-600 hover:bg-green-700 text-white"
        >
          <Check className="mr-1.5 h-4 w-4" />
          Approve
        </Button>
        <Button
          variant="outline"
          onClick={() => onClick(false)}
          disabled={disabled || resume.isPending}
        >
          <X className="mr-1.5 h-4 w-4" />
          Reject
        </Button>
        {resume.isPending && (
          <span className="self-center text-xs text-muted-foreground">
            Resuming…
          </span>
        )}
      </div>
      {error && (
        <p className="text-xs text-destructive">
          {error}
        </p>
      )}
    </div>
  );
}
