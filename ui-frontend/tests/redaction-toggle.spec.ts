/**
 * Captures the v1.14 "Mask secrets" toggle on the trace detail page —
 * off state (raw secret visible) → on state (secret replaced with
 * [REDACTED]) — so docs/security.md can embed visual proof.
 *
 * Driven by ``scripts/capture-redaction-toggle-screenshots.sh``, which
 * boots a fresh UI server seeded with one trace containing known fake
 * secrets and a ``RedactionPolicy(mode="both")`` installed so the
 * toggle has something to do.
 *
 * Output paths:
 *   docs/ui/screenshots/0_2-redaction-toggle-off.png
 *   docs/ui/screenshots/0_2-redaction-toggle-on.png
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

// The seed script writes this trace and these specific values into
// local.db before the server starts. Keep both in sync.
const TRACE_ID = "demo-redact-trace";
const RAW_SECRET = "sk-PROD12345678901234567890ABCDEFGH";

test.describe.configure({ mode: "serial" });

// SpanInspector partitions attributes by key: ``gen_ai.response.content``
// and ``agent.output`` live on the Output tab (see OUTPUT_KEYS in
// ui-frontend/src/components/traces/SpanInspector.tsx). Click that tab
// so the raw / redacted secret lands in the DOM where Playwright can
// assert against it.
async function openOutputTab(page: import("@playwright/test").Page) {
  await page.goto(`/traces/${TRACE_ID}`);
  await page
    .getByRole("switch", { name: /Toggle trace attribute redaction/i })
    .waitFor();
  await page.getByRole("tab", { name: /^Output$/ }).click();
}

test("0_2 — redaction toggle OFF shows raw secret", async ({ page }) => {
  await openOutputTab(page);
  await expect(page.getByText(RAW_SECRET, { exact: false }).first()).toBeVisible(
    { timeout: 5_000 }
  );
  await page.screenshot(SHOT("0_2-redaction-toggle-off"));
});

test("0_2 — redaction toggle ON masks the secret to [REDACTED]", async ({
  page,
}) => {
  await openOutputTab(page);
  await page
    .getByRole("switch", { name: /Toggle trace attribute redaction/i })
    .click();
  // React Query re-keys on ``redact`` and the new fetch remounts
  // SpanInspector, which resets its Tabs state to "Input". Click
  // Output again so the redacted ``gen_ai.response.content`` is on
  // screen for the screenshot.
  await page.getByRole("tab", { name: /^Output$/ }).click();
  await expect(page.getByText(RAW_SECRET, { exact: false })).toHaveCount(0, {
    timeout: 5_000,
  });
  await expect(
    page.getByText("[REDACTED]", { exact: false }).first()
  ).toBeVisible();
  await page.screenshot(SHOT("0_2-redaction-toggle-on"));
});
