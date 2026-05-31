import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Live capture spec for `enable_otel_capture()`. A real OpenInference-convention
 * span is seeded into a real local.db through the production write path (see
 * scripts/seed_foreign_otel.py with normalization on), the UI server is launched
 * against it, and we assert the foreign span renders richly in the real browser:
 * it shows up in the Traces list, and the detail page surfaces Tokens / Cost and
 * the normalized Input + Output content.
 *
 * Run via scripts/capture-foreign-otel-screenshots.sh, which seeds the DB,
 * starts the server, and passes PLAYWRIGHT_BASE_URL + FOREIGN_TRACE_ID.
 *
 * NOTE: the shipped UI has no dedicated "framework badge" component, and the
 * WorkflowBadge only knows agent/chain/swarm/supervisor — so a foreign "llm"
 * span shows no special chip. The framework slug + runner type are verified at
 * the API layer instead (tests/test_ui_foreign_span_render.py). Here we assert
 * the rich rendering that the shipped UI *does* show for a foreign span:
 * Tokens + Cost in the summary bar, and the normalized Input/Output content.
 */

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SCREENSHOT_DIR = path.resolve(__dirname, "../../docs/ui/screenshots");
mkdirSync(SCREENSHOT_DIR, { recursive: true });
const SHOT = (name: string) => ({
  path: path.join(SCREENSHOT_DIR, `${name}.png`),
  fullPage: true,
});

const TRACE_ID = process.env.FOREIGN_TRACE_ID ?? "";

test.describe.configure({ mode: "serial" });

test("foreign-otel-1 — captured foreign span appears in the Traces list", async ({ page }) => {
  await page.goto("/traces");
  // The seeded foreign span surfaces as a trace named after its root span.
  await expect(page.getByText("ChatOpenAI").first()).toBeVisible({ timeout: 15000 });
  await page.waitForTimeout(300);
  await page.screenshot(SHOT("foreign-otel-1-traces-list"));
});

test("foreign-otel-2 — trace detail renders tokens, cost, input & output", async ({ page }) => {
  test.skip(!TRACE_ID, "FOREIGN_TRACE_ID not provided by the harness");
  await page.goto(`/traces/${TRACE_ID}`);

  // Summary bar: Tokens + Cost stats are present (rolled up from the
  // normalized gen_ai.usage.* + computed from gen_ai.request.model). These are
  // blank for a foreign span without the capture normalizer.
  await expect(page.getByText("Tokens", { exact: true })).toBeVisible({ timeout: 15000 });
  await expect(page.getByText("Cost", { exact: true })).toBeVisible();
  // The 1,550 total tokens (1200 + 350) render via formatTokens → "1.6k".
  await expect(page.getByText("1.6k").first()).toBeVisible();

  // The SpanInspector renders from a SEPARATE /spans query — wait for the span
  // tree to load (root auto-selected) before asserting the inspector content.
  await expect(page.getByText("ChatOpenAI").first()).toBeVisible({ timeout: 15000 });

  // Input tab is the default — prompt content from the normalized
  // gen_ai.request.messages (SpanInspector's INPUT_KEYS).
  await page.getByRole("tab", { name: "Input" }).click();
  await expect(page.getByText(/quarterly earnings report/i).first()).toBeVisible({
    timeout: 15000,
  });

  // Output tab — completion content from the normalized gen_ai.response.content
  // (SpanInspector's OUTPUT_KEYS).
  await page.getByRole("tab", { name: "Output" }).click();
  await expect(page.getByText(/Revenue grew 12 percent/i).first()).toBeVisible({
    timeout: 15000,
  });

  await page.waitForTimeout(300);
  await page.screenshot(SHOT("foreign-otel-2-trace-detail"));
});
