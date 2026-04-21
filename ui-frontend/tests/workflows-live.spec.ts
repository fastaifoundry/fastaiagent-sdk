import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Captures the Workflows directory + a detail page against the LIVE local.db
 * on :7842 after running examples/39_workflows_demo.py. Separate from the
 * snapshot-seeded screenshots.spec.ts because those use synthetic fixtures;
 * these show real OpenAI-backed runs so the docs have authentic visuals.
 *
 * Run: PLAYWRIGHT_BASE_URL=http://127.0.0.1:7842 npx playwright test tests/workflows-live.spec.ts
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

test("21 — Workflows directory (live)", async ({ page }) => {
  await page.goto("/workflows");
  await expect(page.getByRole("heading", { name: /Workflows/i })).toBeVisible();
  await expect(page.getByText("content-pipeline").first()).toBeVisible();
  await page.waitForTimeout(300);
  await page.screenshot(SHOT("21-workflows"));
});

test("22 — Workflow detail — chain content-pipeline (live)", async ({ page }) => {
  await page.goto("/workflows/chain/content-pipeline");
  await expect(
    page.getByRole("heading", { name: /content-pipeline/i })
  ).toBeVisible();
  await expect(page.getByText(/Avg latency/i)).toBeVisible();
  // Wait for the filtered traces table to populate.
  await page.waitForTimeout(500);
  await page.screenshot(SHOT("22-workflow-detail"));
});

test("23 — Workflows filtered to swarms (live)", async ({ page }) => {
  await page.goto("/workflows");
  await page.getByRole("tab", { name: /^Swarms$/i }).click();
  await expect(page.getByText("support-triage").first()).toBeVisible();
  await page.waitForTimeout(300);
  await page.screenshot(SHOT("23-workflows-swarms"));
});
