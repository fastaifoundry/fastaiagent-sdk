/**
 * Sprint 2 — Local UI feature evidence captures.
 *
 * Drives the Prompt Playground in a real browser against the live FastAPI
 * server started by ``scripts/capture-sprint2-screenshots.sh`` and writes
 * PNGs into ``docs/ui/screenshots/sprint2-<n>-<feature>.png`` for the docs
 * to embed.
 *
 * The streaming run shot is gated on ``OPENAI_API_KEY`` because it requires
 * a real LLM call. The static-layout shots run without a key.
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

// ---------------------------------------------------------------------------
// Sprint 2 / Feature 1 — Prompt Playground
// ---------------------------------------------------------------------------

test("sprint2-1 — playground empty state", async ({ page }) => {
  await page.goto("/playground");
  await expect(
    page.getByRole("heading", { name: /^Playground$/i })
  ).toBeVisible();

  // Two-panel layout always renders, even before a prompt is selected.
  // Section titles are CardTitles (not the sidebar nav header), so match
  // exactly to disambiguate "// PROMPT" from "// PROMPT REGISTRY".
  await expect(page.getByText("// PROMPT", { exact: true })).toBeVisible();
  await expect(page.getByText("// MODEL", { exact: true })).toBeVisible();
  await expect(page.getByText("// RESPONSE", { exact: true })).toBeVisible();
  await expect(page.getByText("// HISTORY", { exact: true })).toBeVisible();

  // Sidebar shows the new Playground item under PROMPT REGISTRY.
  await expect(
    page.getByRole("link", { name: /^Playground$/i }).first()
  ).toBeVisible();

  await page.waitForTimeout(300);
  await page.screenshot(SHOT("sprint2-1-playground-empty"));
});

test("sprint2-2 — selecting a prompt populates variables", async ({ page }) => {
  await page.goto("/playground");

  // Open the prompt selector and pick the seeded support-greeting prompt.
  await page
    .getByRole("combobox")
    .first()
    .click();
  await page.getByRole("option", { name: /support-greeting/i }).click();

  // Variable inputs auto-render from {{name}} placeholders.
  await expect(page.getByLabel("{{company}}")).toBeVisible();
  await expect(page.getByLabel("{{customer_name}}")).toBeVisible();
  await expect(page.getByLabel("{{topic}}")).toBeVisible();

  // Fill them and confirm the resolved preview updates.
  await page.getByLabel("{{company}}").fill("Acme Co");
  await page.getByLabel("{{customer_name}}").fill("Alice");
  await page.getByLabel("{{topic}}").fill("refunds");
  await expect(page.getByText(/Acme Co/).first()).toBeVisible();

  await page.waitForTimeout(300);
  await page.screenshot(SHOT("sprint2-2-playground-prompt-selected"));
});

const HAS_OPENAI = !!process.env.OPENAI_API_KEY;

test.skip(!HAS_OPENAI, "OPENAI_API_KEY not set — skipping live LLM run");

test("sprint2-3 — running streams a real response with metadata", async ({
  page,
}) => {
  await page.goto("/playground");

  await page.getByRole("combobox").first().click();
  await page.getByRole("option", { name: /support-greeting/i }).click();
  await page.getByLabel("{{company}}").fill("Acme Co");
  await page.getByLabel("{{customer_name}}").fill("Alice");
  await page.getByLabel("{{topic}}").fill("refunds");

  // Click Run and wait for the metadata bar (latency badge) to appear, which
  // only renders after the stream's "done" event arrives.
  await page.getByRole("button", { name: /^Run$/ }).click();
  await expect(page.getByText(/openai\/gpt-4o-mini/i)).toBeVisible({
    timeout: 30_000,
  });

  // Trace link surfaces the trace_id for click-through verification.
  // Match exactly to disambiguate from the sidebar's "Traces" nav link.
  await expect(
    page.getByRole("link", { name: "trace", exact: true })
  ).toBeVisible();

  await page.waitForTimeout(500);
  await page.screenshot(SHOT("sprint2-3-playground-streamed-response"));
});

// ---------------------------------------------------------------------------
// Sprint 2 / Feature 2 — Agent Dependency Graph
// ---------------------------------------------------------------------------

test("sprint2-4 — supervisor dependency graph with worker subtrees", async ({
  page,
}) => {
  await page.goto("/agents/planner");

  // Wait for the agent header so we know the page rendered.
  await expect(
    page.getByRole("heading", { name: /^planner$/ })
  ).toBeVisible({ timeout: 10_000 });

  // Switch to the new Dependencies tab.
  await page.getByRole("tab", { name: /^Dependencies$/ }).click();

  // The React Flow canvas renders inside a tagged container.
  const canvas = page.locator('[data-testid="agent-dependency-graph"]');
  await expect(canvas).toBeVisible({ timeout: 15_000 });

  // Centre node + each worker.
  await expect(page.locator('[data-node-kind="agent"]')).toBeVisible();
  await expect(page.locator('[data-node-kind="worker"]').first()).toBeVisible();

  // Worker subtree spreads tools / KBs.
  await expect(page.locator('[data-node-kind="tool"]').first()).toBeVisible();
  await expect(page.locator('[data-node-kind="kb"]').first()).toBeVisible();

  // Let React Flow settle before capturing the shot.
  await page.waitForTimeout(800);
  await page.screenshot(SHOT("sprint2-4-agent-dependency-graph"));
});

test("sprint2-5 — single worker dependency view", async ({ page }) => {
  await page.goto("/agents/researcher");
  await expect(
    page.getByRole("heading", { name: /^researcher$/ })
  ).toBeVisible({ timeout: 10_000 });

  await page.getByRole("tab", { name: /^Dependencies$/ }).click();
  const canvas = page.locator('[data-testid="agent-dependency-graph"]');
  await expect(canvas).toBeVisible({ timeout: 15_000 });

  // Worker has model + at least one tool / kb.
  await expect(page.locator('[data-node-kind="model"]').first()).toBeVisible();
  await expect(page.locator('[data-node-kind="tool"]').first()).toBeVisible();

  await page.waitForTimeout(600);
  await page.screenshot(SHOT("sprint2-5-agent-dependency-worker"));
});

// ---------------------------------------------------------------------------
// Sprint 2 / Feature 3 — Guardrail Event Detail
// ---------------------------------------------------------------------------

const BLOCKED_EVENT_ID = process.env.SPRINT2_BLOCKED_EVENT_ID;
const FILTERED_EVENT_ID = process.env.SPRINT2_FILTERED_EVENT_ID;

test("sprint2-6 — guardrails list shows new filters and clickable rows", async ({
  page,
}) => {
  await page.goto("/guardrails");
  // Filters bar is rendered.
  await expect(
    page.getByRole("combobox").nth(0)
  ).toBeVisible(); // outcome select
  // Demo seed inserts 3 events.
  await expect(page.getByText(/no_pii/).first()).toBeVisible({
    timeout: 10_000,
  });
  await expect(page.getByText(/email_redactor/).first()).toBeVisible();
  await expect(page.getByText(/toxicity_check/).first()).toBeVisible();
  await page.waitForTimeout(400);
  await page.screenshot(SHOT("sprint2-6-guardrails-list-filters"));
});

test.skip(
  !BLOCKED_EVENT_ID,
  "SPRINT2_BLOCKED_EVENT_ID not set — capture script wires it",
);

test("sprint2-7 — event detail page renders three panels (blocked)", async ({
  page,
}) => {
  await page.goto(`/guardrail-events/${BLOCKED_EVENT_ID}`);
  await expect(
    page.getByRole("heading", { name: /no_pii/i })
  ).toBeVisible({ timeout: 10_000 });
  await expect(page.locator('[data-testid="panel-trigger"]')).toBeVisible();
  await expect(page.locator('[data-testid="panel-rule"]')).toBeVisible();
  await expect(page.locator('[data-testid="panel-outcome"]')).toBeVisible();
  // Triggering content shows up in panel 1.
  await expect(page.getByText(/alice@example\.com/).first()).toBeVisible();
  await page.waitForTimeout(500);
  await page.screenshot(SHOT("sprint2-7-guardrail-detail-blocked"));
});

test("sprint2-8 — filtered event shows before/after diff", async ({ page }) => {
  if (!FILTERED_EVENT_ID) test.skip();
  await page.goto(`/guardrail-events/${FILTERED_EVENT_ID}`);
  await expect(page.getByText(/before/i).first()).toBeVisible({
    timeout: 10_000,
  });
  await expect(page.getByText(/after/i).first()).toBeVisible();
  await expect(page.getByText(/REDACTED/).first()).toBeVisible();
  await page.waitForTimeout(500);
  await page.screenshot(SHOT("sprint2-8-guardrail-detail-filtered"));
});

test("sprint2-9 — false-positive toggle persists across refresh", async ({
  page,
}) => {
  if (!BLOCKED_EVENT_ID) test.skip();
  await page.goto(`/guardrail-events/${BLOCKED_EVENT_ID}`);
  // Click the FP button. Whatever its current state, the click flips it.
  const btn = page.getByTestId("mark-false-positive-button");
  await expect(btn).toBeVisible();
  const labelBefore = (await btn.textContent()) ?? "";
  await btn.click();
  await page.waitForTimeout(400);
  const labelAfter = (await btn.textContent()) ?? "";
  expect(labelAfter).not.toBe(labelBefore);

  // Refresh — flag still stuck.
  await page.reload();
  await expect(page.getByTestId("mark-false-positive-button")).toBeVisible();
  const labelAfterReload = await page
    .getByTestId("mark-false-positive-button")
    .textContent();
  expect(labelAfterReload).toBe(labelAfter);

  await page.waitForTimeout(400);
  await page.screenshot(SHOT("sprint2-9-guardrail-false-positive"));

  // Toggle back so reruns of this spec are idempotent.
  await page.getByTestId("mark-false-positive-button").click();
});
