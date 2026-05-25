/**
 * Captures the trace-detail UI for the regression-from-trace template's
 * failing and fixed runs, so docs/flagships/regression-from-trace.md
 * (and the example README) can embed visual proof.
 *
 * Driven by ``scripts/capture-regression-from-trace-screenshots.sh``,
 * which runs ``capture.py`` then ``fix.py`` against a tmpdir
 * local.db, boots the UI against that DB, and runs this spec.
 *
 * The spec reads the two trace IDs from
 * ``FAILING_TRACE_ID`` / ``FIXED_TRACE_ID`` env vars set by the
 * orchestrator.
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

const FAILING_TRACE_ID = process.env.FAILING_TRACE_ID ?? "";
const FIXED_TRACE_ID = process.env.FIXED_TRACE_ID ?? "";

test.describe.configure({ mode: "serial" });

test("0_3 — failing trace shows wrong agent output for ORD-999", async ({
  page,
}) => {
  test.skip(!FAILING_TRACE_ID, "FAILING_TRACE_ID env var not set");
  await page.goto(`/traces/${FAILING_TRACE_ID}`);
  await page.getByRole("tab", { name: /^Output$/ }).click();
  // The buggy run confidently reports a delivery for an order that
  // doesn't exist. The LLM paraphrases the date and item differently
  // across runs — "delivered" is the LLM-invariant tell.
  await expect(page.getByText(/delivered/i).first()).toBeVisible({
    timeout: 5_000,
  });
  await page.screenshot(SHOT("0_3-failing-trace"));
});

test("0_3 — fixed trace shows the corrected not-found reply", async ({
  page,
}) => {
  test.skip(!FIXED_TRACE_ID, "FIXED_TRACE_ID env var not set");
  await page.goto(`/traces/${FIXED_TRACE_ID}`);
  await page.getByRole("tab", { name: /^Output$/ }).click();
  // After the tool-override rerun, the agent correctly reports
  // "not found" for ORD-999.
  await expect(page.getByText(/not found/i).first()).toBeVisible({
    timeout: 5_000,
  });
  await page.screenshot(SHOT("0_3-fixed-trace"));
});
