/**
 * Temporary stubs for routes not yet built out.
 *
 * Each slice (7–11) replaces the matching page with real content while
 * keeping the same route path. These exist so the sidebar + routing shell
 * is functional from Slice 6.
 */

import { Construction } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { EmptyState } from "@/components/shared/EmptyState";

function Stub({ title, description }: { title: string; description: string }) {
  return (
    <div className="space-y-6">
      <PageHeader title={title} description={description} />
      <EmptyState
        icon={Construction}
        title="Coming soon"
        description="This surface is being built — reconnect after the next SDK release."
      />
    </div>
  );
}

export const CompareTracesPage = () => (
  <Stub title="Compare traces" description="Side-by-side diff of two traces." />
);
