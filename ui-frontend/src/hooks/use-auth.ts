import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/auth-store";

interface AuthStatus {
  authenticated: boolean;
  username: string | null;
  no_auth: boolean;
}

/**
 * Kept in a single place so every page can trust `useAuthStore` without
 * redundantly hitting `/api/auth/status`.
 */
export function useAuth() {
  const setStatus = useAuthStore((s) => s.setStatus);

  const query = useQuery({
    queryKey: ["auth", "status"],
    queryFn: () => api.get<AuthStatus>("/auth/status"),
    staleTime: 60_000,
  });

  useEffect(() => {
    if (query.data) setStatus(query.data);
  }, [query.data, setStatus]);

  return query;
}
