import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FastAIAgentLogo } from "@/components/brand/FastAIAgentLogo";
import { api, ApiError } from "@/lib/api";

export function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const login = useMutation({
    mutationFn: (payload: { username: string; password: string }) =>
      api.post<{ status: string; username: string }>("/auth/login", payload),
    onSuccess: () => {
      toast.success("Welcome back");
      queryClient.invalidateQueries({ queryKey: ["auth", "status"] });
      navigate("/", { replace: true });
    },
    onError: (e) => {
      if (e instanceof ApiError) {
        toast.error(e.status === 401 ? "Invalid username or password" : e.message);
      } else {
        toast.error("Login failed");
      }
    },
  });

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <Card className="w-full max-w-sm">
        <CardHeader className="items-center text-center">
          <FastAIAgentLogo className="mb-3 h-12 w-12 rounded" variant="favicon" />
          <CardTitle className="text-2xl">FastAIAgent</CardTitle>
          <CardDescription>Local UI — sign in to continue.</CardDescription>
        </CardHeader>
        <CardContent>
          <form
            className="space-y-4"
            onSubmit={(e) => {
              e.preventDefault();
              login.mutate({ username, password });
            }}
          >
            <div className="space-y-1.5">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <Button type="submit" className="w-full" disabled={login.isPending}>
              {login.isPending ? "Signing in…" : "Sign in"}
            </Button>
            <p className="text-xs text-muted-foreground text-center pt-1">
              Forgot password? Delete <code>.fastaiagent/auth.json</code> and run{" "}
              <code>fastaiagent ui</code> again.
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
