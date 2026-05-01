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
      partialize: (s) => ({
        username: s.username,
        authenticated: s.authenticated,
        noAuth: s.noAuth,
        projectId: s.projectId,
      }),
    }
  )
);
