import { create } from "zustand";
import { persist } from "zustand/middleware";

interface AuthState {
  username: string | null;
  authenticated: boolean;
  noAuth: boolean;
  projectId: string;
  setStatus: (status: {
    authenticated: boolean;
    username: string | null;
    no_auth: boolean;
    project_id?: string;
  }) => void;
  clear: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      username: null,
      authenticated: false,
      noAuth: false,
      projectId: "",
      setStatus: ({ authenticated, username, no_auth, project_id }) =>
        set({
          authenticated,
          username,
          noAuth: no_auth,
          projectId: project_id ?? "",
        }),
      clear: () =>
        set({ username: null, authenticated: false, projectId: "" }),
    }),
    {
      name: "fastaiagent-auth",
      // security_review_1.md M12: do NOT persist ``authenticated`` —
      // the only authoritative source for "is this user logged in" is
      // the httpOnly session cookie validated by ``/api/auth/status``.
      // Persisting the flag in localStorage made it forgeable by any
      // XSS payload (or browser-extension content script). The store
      // still caches harmless display fields (username, project)
      // across reloads so the chrome doesn't flicker; ``authenticated``
      // resets to ``false`` on every page load and is filled in by the
      // first ``/api/auth/status`` round-trip.
      partialize: (s) => ({
        username: s.username,
        noAuth: s.noAuth,
        projectId: s.projectId,
      }),
    }
  )
);
