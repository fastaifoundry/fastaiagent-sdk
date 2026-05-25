/**
 * Captures the v1.14.1 "Rerun with modifications" dialog (formerly
 * "Rerun from this step") so docs/replay/index.md can embed the
 * corrected UX and the screenshot regression catches anyone reverting
 * the copy.
 *
 * Driven by ``scripts/capture-replay-fork-dialog-screenshot.sh``,
 * which boots a UI seeded with a single agent trace, navigates to the
 * trace's Replay page, clicks "Fork this step" to open the dialog,
 * and captures the dialog state with the new copy and the new "Tool
 * name + JSON" two-field shape from the Tool tab.
 *
 * Output:
 *   docs/ui/screenshots/0_3_audit-rerun-dialog.png
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

const TRACE_ID = process.env.REPLAY_FORK_TRACE_ID ?? "replay-dialog-trace";

test.describe.configure({ mode: "serial" });

test("0_3_audit — rerun-with-modifications dialog opens with corrected copy", async ({
  page,
}) => {
  await page.goto(`/traces/${TRACE_ID}/replay`);

  // Open the fork dialog. The Replay page lists steps; "Fork here"
  // is the per-row trigger.
  const forkButton = page.getByRole("button", { name: /^Fork here$/ }).first();
  await forkButton.waitFor({ timeout: 5_000 });
  await forkButton.click();

  // v1.14.1: title was renamed from "Fork and rerun" to
  // "Fork and rerun with modifications" so the copy matches what
  // the rerun actually does (full re-execution, not mid-trace resume).
  await expect(
    page.getByRole("heading", { name: /Fork and rerun with modifications/i })
  ).toBeVisible();

  // The submit button copy moved from "Rerun from this step" (misleading)
  // to "Rerun with modifications" (accurate).
  await expect(
    page.getByRole("button", { name: /^Rerun with modifications$/ })
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /^Rerun from this step$/ })
  ).toHaveCount(0);

  // Switch to the Tool tab to capture the new two-field shape
  // ("Tool name" + "Tool response override (JSON)") instead of the
  // pre-v1.14.1 single-textarea that silently did nothing.
  await page.getByRole("tab", { name: /^Tool/ }).click();
  await expect(page.getByLabel("Tool name")).toBeVisible();
  await expect(page.getByLabel(/Tool response override/)).toBeVisible();

  await page.screenshot(SHOT("0_3_audit-rerun-dialog"));
});
