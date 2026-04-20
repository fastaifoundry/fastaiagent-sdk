import { create } from "zustand";
import { persist } from "zustand/middleware";

interface AuthState {
  username: string | null;
  authenticated: boolean;
  noAuth: boolean;
  setStatus: (status: { authenticated: boolean; username: string | null; no_auth: boolean }) => void;
  clear: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      username: null,
      authenticated: false,
      noAuth: false,
      setStatus: ({ authenticated, username, no_auth }) =>
        set({ authenticated, username, noAuth: no_auth }),
      clear: () => set({ username: null, authenticated: false }),
    }),
    {
      name: "fastaiagent-auth",
      partialize: (s) => ({ username: s.username, authenticated: s.authenticated, noAuth: s.noAuth }),
    }
  )
);
