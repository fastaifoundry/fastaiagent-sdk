/**
 * Captures the Local UI surfaces after the full example sweep so the
 * commit message can show what every page looks like with real data.
 *
 * Driven by ``scripts/capture-example-sweep-screenshots.sh`` — that
 * script boots the production-built UI server against the repo's
 * ``.fastaiagent/local.db`` (which holds the spans / evals / prompts
 * the example sweep produced) and runs this spec.
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

test("examples-1 — overview shows 24h activity from the sweep", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Home/i })).toBeVisible();
  await expect(page.getByText(/Traces \(24h\)/i)).toBeVisible();
  await page.waitForTimeout(400);
  await page.screenshot(SHOT("examples-1-overview"));
});

test("examples-2 — traces list populated", async ({ page }) => {
  await page.goto("/traces");
  await expect(page.getByRole("heading", { name: /Traces/i })).toBeVisible();
  // After the sweep, agent.* / chain.* / swarm.* rows must all be present.
  await page.waitForTimeout(400);
  await page.screenshot(SHOT("examples-2-traces"));
});

test("examples-3 — workflows directory shows chains, swarms, supervisors", async ({
  page,
}) => {
  await page.goto("/workflows");
  await expect(page.getByRole("heading", { name: /Workflows/i })).toBeVisible();
  await page.waitForTimeout(400);
  await page.screenshot(SHOT("examples-3-workflows"));
});

test("examples-4 — agents directory", async ({ page }) => {
  await page.goto("/agents");
  await expect(page.getByRole("heading", { name: /Agents/i })).toBeVisible();
  await page.waitForTimeout(400);
  await page.screenshot(SHOT("examples-4-agents"));
});

test("examples-5 — analytics", async ({ page }) => {
  await page.goto("/analytics");
  await expect(page.getByRole("heading", { name: /Analytics/i })).toBeVisible();
  await page.waitForTimeout(700);
  await page.screenshot(SHOT("examples-5-analytics"));
});

test("examples-6 — eval runs list", async ({ page }) => {
  await page.goto("/evals");
  await expect(page.getByRole("heading", { name: /Eval runs/i })).toBeVisible();
  await page.waitForTimeout(400);
  await page.screenshot(SHOT("examples-6-evals"));
});

test("examples-7 — prompts list", async ({ page }) => {
  await page.goto("/prompts");
  await expect(page.getByRole("heading", { name: /Prompts/i })).toBeVisible();
  await page.waitForTimeout(300);
  await page.screenshot(SHOT("examples-7-prompts"));
});

test("examples-8 — guardrail events", async ({ page }) => {
  await page.goto("/guardrails");
  await expect(
    page.getByRole("heading", { name: /Guardrail events/i })
  ).toBeVisible();
  await page.waitForTimeout(300);
  await page.screenshot(SHOT("examples-8-guardrails"));
});

test("examples-9 — header breadcrumb shows the active project", async ({
  page,
}) => {
  await page.goto("/");
  // The orchestrator boots the server with project_id="fastaiagent-sdk"
  // so the breadcrumb reads ``Local UI // fastaiagent-sdk // auth disabled``.
  await expect(page.getByTestId("project-id")).toHaveText("fastaiagent-sdk");
  await page.waitForTimeout(200);
  await page.screenshot(SHOT("examples-9-breadcrumb"));
});
