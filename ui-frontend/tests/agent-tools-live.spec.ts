import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Live screenshot for the Agent Detail Tools section. Requires
 * examples/41_agent_tools.py to have run first (produces the
 * "tool-curator" agent with a mixed-origin toolkit).
 *
 * Run:
 *   PLAYWRIGHT_BASE_URL=http://127.0.0.1:7842 \
 *     npx playwright test tests/agent-tools-live.spec.ts
 */

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SCREENSHOT_DIR = path.resolve(__dirname, "../../docs/ui/screenshots");
mkdirSync(SCREENSHOT_DIR, { recursive: true });
const SHOT = (name: string) => ({
  path: path.join(SCREENSHOT_DIR, `${name}.png`),
  fullPage: true,
});

test("27 — agent detail with Tools section (live)", async ({ page }) => {
  await page.goto("/agents/tool-curator");
  await expect(
    page.getByRole("heading", { name: /tool-curator/i })
  ).toBeVisible();
  await expect(page.getByText(/Tools/).first()).toBeVisible();
  // Any one of the three tools the example registers.
  await expect(page.getByText("c_to_f").first()).toBeVisible();
  await expect(page.getByText("count_punctuation").first()).toBeVisible();
  await expect(page.getByText("unused").first()).toBeVisible();
  await page.waitForTimeout(400);
  await page.screenshot(SHOT("27-agent-tools"));
});
