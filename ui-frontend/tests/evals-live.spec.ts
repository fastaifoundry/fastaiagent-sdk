import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Live screenshots for the eval pages. Requires examples/40_evals_compare.py
 * to have been run first so ./.fastaiagent/local.db contains real before/after
 * runs. Separate from screenshots.spec.ts because the snapshot seeder only
 * populates synthetic eval fixtures.
 *
 * Run: PLAYWRIGHT_BASE_URL=http://127.0.0.1:7842 npx playwright test tests/evals-live.spec.ts
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

test("24 — eval runs list with cost + latency (live)", async ({ page }) => {
  await page.goto("/evals");
  await expect(page.getByRole("heading", { name: /Eval runs/i })).toBeVisible();
  await expect(page.getByText("echo-bot-v2-strict").first()).toBeVisible();
  await expect(page.getByText("Avg latency")).toBeVisible();
  await page.waitForTimeout(300);
  await page.screenshot(SHOT("24-evals-list"));
});

test("25 — eval run detail with scorer chips + filters (live)", async ({ page }) => {
  // Use the first row in the list so we don't hardcode a run_id.
  await page.goto("/evals");
  await page.getByText("echo-bot-v2-strict").first().click();
  await expect(
    page.getByRole("heading", { name: /echo-bot-v2-strict/i })
  ).toBeVisible();
  await expect(page.getByText(/Scorers/i).first()).toBeVisible();
  // Expand the first case to show the diff inside the table.
  const firstExpand = page.getByRole("button", { name: /Expand case/i }).first();
  await firstExpand.click();
  await page.waitForTimeout(400);
  await page.screenshot(SHOT("25-eval-detail"));
});

test("26 — eval compare view with improvements highlighted (live)", async ({
  page,
}) => {
  // Discover the two most recent echo-bot runs via the list API so the
  // screenshot isn't pinned to a specific run_id.
  const res = await page.request.get("/api/evals?agent=echo-bot&page_size=10");
  const body = await res.json();
  const rows = body.rows as Array<{ run_id: string; run_name: string | null }>;
  const runB = rows.find((r) => r.run_name === "echo-bot-v2-strict")?.run_id;
  const runA = rows.find((r) => r.run_name === "echo-bot-v1-vague")?.run_id;
  if (!runA || !runB) {
    throw new Error(
      "Expected examples/40_evals_compare.py runs to be present in local.db"
    );
  }

  await page.goto(`/evals/compare?a=${runA}&b=${runB}`);
  await expect(
    page.getByRole("heading", { name: /Compare eval runs/i })
  ).toBeVisible();
  await expect(page.getByText(/failed in A, passed in B/i)).toBeVisible();
  await expect(page.getByText(/net improvement/i)).toBeVisible();
  await page.waitForTimeout(500);
  await page.screenshot(SHOT("26-eval-compare"));
});
