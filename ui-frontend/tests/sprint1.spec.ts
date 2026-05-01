/**
 * Sprint 1 — Local UI feature evidence captures.
 *
 * Each test drives a Sprint 1 feature in a real browser against the live
 * FastAPI server started by ``scripts/capture-sprint1-screenshots.sh``,
 * then writes a PNG into ``docs/ui/screenshots/sprint1-<n>-<feature>.png``
 * that the docs and example READMEs embed as evidence.
 *
 * Independent of ``screenshots.spec.ts`` — that one runs against the
 * snapshot DB without registered runners, which is fine for trace-detail
 * surfaces but doesn't have data the topology view needs.
 */
import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SCREENSHOT_DIR = path.resolve(__dirname, "../../docs/ui/screenshots");
mkdirSync(SCREENSHOT_DIR, { recursive: true });

const SHOT = (name: string) => ({
  path: path.join(SCREENSHOT_DIR, `${name}.png`),
  fullPage: true,
});

test.describe.configure({ mode: "serial" });

test("sprint1-1 — workflow topology renders for a registered chain", async ({
  page,
}) => {
  await page.goto("/workflows/chain/refund-flow");

  // Wait for the page header so we know routing landed.
  await expect(
    page.getByRole("heading", { name: /refund-flow/i })
  ).toBeVisible();

  // The topology canvas is rendered into a sibling that lazy-loads its
  // chunk; wait for the canvas.
  const canvas = page.locator('[data-testid="workflow-topology"]');
  await expect(canvas).toBeVisible({ timeout: 15000 });

  // Each node from examples/47 should appear with its type tag.
  await expect(
    page.locator('[data-node-type="agent"]').first()
  ).toBeVisible();
  await expect(
    page.locator('[data-node-type="hitl"]')
  ).toBeVisible();

  // Node labels match the example chain.
  await expect(page.getByText("researcher").first()).toBeVisible();
  await expect(page.getByText("Manager approval")).toBeVisible();
  await expect(page.getByText("notifier").first()).toBeVisible();

  // Layout toggle is present in non-compact mode.
  await expect(
    page.getByRole("button", { name: /Horizontal/i })
  ).toBeVisible();

  // Give the React Flow renderer a moment to settle (animation + edge paths).
  await page.waitForTimeout(600);
  await page.screenshot(SHOT("sprint1-1-workflow-topology"));
});

test("sprint1-1b — clicking a node opens the detail panel", async ({ page }) => {
  await page.goto("/workflows/chain/refund-flow");
  const canvas = page.locator('[data-testid="workflow-topology"]');
  await expect(canvas).toBeVisible({ timeout: 15000 });

  await page.locator('[data-node-type="agent"]').first().click();
  await expect(page.getByText(/^Agent$/)).toBeVisible();
  await page.waitForTimeout(300);
  await page.screenshot(SHOT("sprint1-1b-workflow-node-detail"));
});
