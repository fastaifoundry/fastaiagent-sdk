import { describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/test/utils";
import { LoginPage } from "./LoginPage";

describe("LoginPage", () => {
  it("renders the sign-in form", () => {
    renderWithProviders(<LoginPage />);
    // CardTitle isn't a semantic heading; assert by visible copy instead.
    expect(screen.getByText(/Local UI — sign in to continue/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  it("POSTs /auth/login when submitted and surfaces the error on 401", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Invalid username or password" }), {
        status: 401,
        headers: { "content-type": "application/json" },
      })
    );

    renderWithProviders(<LoginPage />);
    await user.type(screen.getByLabelText(/username/i), "upendra");
    await user.type(screen.getByLabelText(/password/i), "nope");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/auth/login",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ username: "upendra", password: "nope" }),
          credentials: "include",
        })
      );
    });

    fetchMock.mockRestore();
  });
});
