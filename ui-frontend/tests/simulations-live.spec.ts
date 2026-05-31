import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Live screenshots for the Simulations pages. Requires
 * `python tests/seed_simulations.py <db>` to have been run first so the
 * server's local.db contains a real simulation run.
 *
 * Run: PLAYWRIGHT_BASE_URL=http://127.0.0.1:7843 npx playwright test tests/simulations-live.spec.ts
 */

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SCREENSHOT_DIR = path.resolve(__dirname, "../../docs/ui/screenshots");
mkdirSync(SCREENSHOT_DIR, { recursive: true });
const SHOT = (name: string) => ({
  path: path.join(SCREENSHOT_DIR, `${name}.png`),
  fullPage: true,
});

test.describe.configure({ mode: "serial" });

test("simulations list renders the seeded run + pass-rate", async ({ page }) => {
  await page.goto("/simulations");
  await expect(page.getByRole("heading", { name: /Simulations/i })).toBeVisible();
  await expect(page.getByText("support-suite-v1").first()).toBeVisible();
  await expect(page.getByText("support-bot").first()).toBeVisible();
  await page.waitForTimeout(300);
  await page.screenshot(SHOT("simulations-list"));
});

test("simulation detail shows transcript + per-criterion chips + view trace", async ({
  page,
}) => {
  await page.goto("/simulations");
  await page.getByText("support-suite-v1").first().click();
  await expect(
    page.getByRole("heading", { name: /support-suite-v1/i })
  ).toBeVisible();

  // Expand the first scenario card to reveal transcript + criterion chips.
  await page.getByText("refund-policy").first().click();
  await expect(page.getByText("Can I get a refund?").first()).toBeVisible();
  await expect(
    page.getByText(/explains the refund policy/i).first()
  ).toBeVisible();
  await expect(page.getByRole("link", { name: /view trace/i }).first()).toBeVisible();
  await page.waitForTimeout(400);
  await page.screenshot(SHOT("simulation-detail"));
});
