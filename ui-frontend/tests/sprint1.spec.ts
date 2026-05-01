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

// ---------------------------------------------------------------------------
// Feature 2 — Multimodal trace rendering
// ---------------------------------------------------------------------------

const MM_TRACE_ID = "mm00000000000000000000000000mm01";

test("sprint1-2 — multimodal trace input renders inline image thumbnail", async ({
  page,
}) => {
  await page.goto(`/traces/${MM_TRACE_ID}`);

  // Span tree loads on the left; pick the LLM span which carries the
  // gen_ai.request.messages attribute with the image content part.
  await expect(
    page.getByRole("heading", { name: /agent\.vision-bot/i })
  ).toBeVisible();
  await page
    .getByRole("button", { name: /llm\.openai\.gpt-4o-mini/ })
    .first()
    .click();

  // The Input tab is selected by default. The MixedContentView should
  // detect the image_url part and render an <img>.
  await expect(
    page.locator('[data-testid="mixed-content-view"]')
  ).toBeVisible({ timeout: 5000 });
  await expect(page.locator("img").first()).toBeVisible();

  // The accompanying text part renders too.
  await expect(
    page.getByText(/What's in this image\?/i).first()
  ).toBeVisible();

  await page.waitForTimeout(400);
  await page.screenshot(SHOT("sprint1-2-multimodal-input"));
});

test("sprint1-2b — attachment gallery + thumbnail render below content parts", async ({
  page,
}) => {
  await page.goto(`/traces/${MM_TRACE_ID}`);

  // The root agent span has fastaiagent.input.media_count=1 → AttachmentGallery
  // fetches the thumbnail tile from the binary endpoint and renders it.
  await page
    .getByRole("button", { name: /agent\.vision-bot/i })
    .first()
    .click();
  await expect(page.locator("img").first()).toBeVisible({ timeout: 5000 });

  await page.waitForTimeout(400);
  await page.screenshot(SHOT("sprint1-2b-multimodal-gallery"));
});

// ---------------------------------------------------------------------------
// Feature 3 — Checkpoint inspector
// ---------------------------------------------------------------------------

const EXEC_ID = "exec-sprint1-mm-00001";

test("sprint1-3 — checkpoint timeline shows completed and interrupted steps", async ({
  page,
}) => {
  await page.goto(`/executions/${EXEC_ID}`);

  await expect(
    page.getByRole("heading", { name: /Execution/i })
  ).toBeVisible();

  // Vertical timeline renders one row per checkpoint, each tagged with status.
  const timeline = page.locator('[data-testid="checkpoint-timeline"]');
  await expect(timeline).toBeVisible();
  await expect(
    page.locator('[data-checkpoint-status="completed"]')
  ).toBeVisible();
  await expect(
    page.locator('[data-checkpoint-status="interrupted"]')
  ).toBeVisible();

  // Recoverable callout fires for the latest interrupted checkpoint.
  await expect(page.getByText(/recoverable/i)).toBeVisible();

  // Idempotency cache section lists the seeded @idempotent rows.
  const idem = page.locator('[data-testid="idempotency-cache"]');
  await expect(idem).toBeVisible();
  await expect(
    page.getByText(/charge_customer:cust_42:500/)
  ).toBeVisible();

  await page.waitForTimeout(300);
  await page.screenshot(SHOT("sprint1-3-checkpoint-timeline"));
});

test("sprint1-3b — expanding two adjacent rows shows the state diff", async ({
  page,
}) => {
  await page.goto(`/executions/${EXEC_ID}`);
  await expect(
    page.locator('[data-testid="checkpoint-timeline"]')
  ).toBeVisible();

  // Expand both checkpoint rows to reveal the diff card below.
  const rows = page.locator(
    '[data-testid="checkpoint-timeline"] li[data-checkpoint-status]'
  );
  await rows.nth(0).getByRole("button").click();
  await rows.nth(1).getByRole("button").click();

  const diff = page.locator('[data-testid="state-diff"]');
  await expect(diff).toBeVisible({ timeout: 3000 });
  await expect(diff.getByText(/Added|Changed/i).first()).toBeVisible();

  // Scroll the diff card into view so the screenshot framing actually
  // shows it (full-page captures still include it, but the focused
  // viewport area is what readers look at first).
  await diff.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  await page.screenshot(SHOT("sprint1-3b-checkpoint-state-diff"));
});
