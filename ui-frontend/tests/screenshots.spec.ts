import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Captures every Local UI surface into docs/ui/screenshots/ so the docs
 * can ship a visual navigation guide that stays in sync with the code.
 *
 * Runs in single-worker mode (playwright.config.ts) — the FastAPI server is
 * a shared singleton that was pre-seeded with fixture data by the
 * orchestration script.
 */

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SCREENSHOT_DIR = path.resolve(__dirname, "../../docs/ui/screenshots");
mkdirSync(SCREENSHOT_DIR, { recursive: true });

const SHOT = (name: string) => ({ path: path.join(SCREENSHOT_DIR, `${name}.png`), fullPage: true });

test.describe.configure({ mode: "serial" });

test("01 — Overview / home", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Home/i })).toBeVisible();
  await expect(page.getByText(/Traces \(24h\)/i)).toBeVisible();
  await page.waitForTimeout(200);
  await page.screenshot(SHOT("01-overview"));
});

test("02 — Traces list", async ({ page }) => {
  await page.goto("/traces");
  await expect(page.getByRole("heading", { name: /Traces/i })).toBeVisible();
  await expect(page.getByText("agent.support-bot")).toBeVisible();
  await page.waitForTimeout(200);
  await page.screenshot(SHOT("02-traces"));
});

test("03 — Trace detail", async ({ page }) => {
  await page.goto("/traces/aaaaaaaaaaaaaaaaaaaaaaaaaaaa1111");
  await expect(page.getByText("agent.support-bot").first()).toBeVisible();
  await expect(page.getByText("tool.lookup_order").first()).toBeVisible();
  await page.waitForTimeout(250);
  await page.screenshot(SHOT("03-trace-detail"));
});

test("04 — Agent Replay", async ({ page }) => {
  await page.goto("/traces/aaaaaaaaaaaaaaaaaaaaaaaaaaaa1111/replay");
  await expect(page.getByRole("heading", { name: /Agent Replay/i })).toBeVisible();
  await expect(page.getByRole("button", { name: /Fork here/i })).toBeVisible();
  await page.waitForTimeout(250);
  await page.screenshot(SHOT("04-agent-replay"));
});

test("05 — Agent Replay fork dialog", async ({ page }) => {
  await page.goto("/traces/aaaaaaaaaaaaaaaaaaaaaaaaaaaa1111/replay");
  await page.getByText("llm.chat").first().click();
  await page.getByRole("button", { name: /Fork here/i }).click();
  await expect(page.getByRole("heading", { name: /Fork and rerun/i })).toBeVisible();
  await page.waitForTimeout(200);
  await page.screenshot(SHOT("05-replay-fork-dialog"));
});

test("06 — Eval runs list", async ({ page }) => {
  await page.goto("/evals");
  await expect(page.getByRole("heading", { name: /Eval runs/i })).toBeVisible();
  await expect(page.getByText("support-smoke")).toBeVisible();
  await page.waitForTimeout(400);
  await page.screenshot(SHOT("06-evals"));
});

test("07 — Eval run detail", async ({ page }) => {
  await page.goto("/evals");
  await page.getByText("support-smoke").click();
  await expect(page.getByText(/Pass rate/i).first()).toBeVisible();
  // The Cases card title is an exact match; the row header also contains
  // "Cases" so scope to the card title.
  await expect(page.getByText("Cases", { exact: true }).first()).toBeVisible();
  await page.waitForTimeout(300);
  await page.screenshot(SHOT("07-eval-detail"));
});

test("08 — Prompts list", async ({ page }) => {
  await page.goto("/prompts");
  await expect(page.getByRole("heading", { name: /Prompts/i })).toBeVisible();
  await expect(page.getByText("ui-demo.support")).toBeVisible();
  await page.waitForTimeout(200);
  await page.screenshot(SHOT("08-prompts"));
});

test("09 — Prompt editor", async ({ page }) => {
  await page.goto("/prompts/ui-demo.support");
  await expect(page.getByRole("heading", { name: /ui-demo.support/i })).toBeVisible();
  await expect(page.getByRole("textbox")).toBeVisible();
  await page.waitForTimeout(300);
  await page.screenshot(SHOT("09-prompt-editor"));
});

test("10 — Guardrail events", async ({ page }) => {
  await page.goto("/guardrails");
  await expect(page.getByRole("heading", { name: /Guardrail events/i })).toBeVisible();
  await expect(page.getByText("no_pii").first()).toBeVisible();
  await page.waitForTimeout(200);
  await page.screenshot(SHOT("10-guardrails"));
});

test("11 — Agents directory", async ({ page }) => {
  await page.goto("/agents");
  await expect(page.getByRole("heading", { name: /Agents/i })).toBeVisible();
  await expect(page.getByText("support-bot").first()).toBeVisible();
  await page.waitForTimeout(200);
  await page.screenshot(SHOT("11-agents"));
});

test("12 — Login page", async ({ page }) => {
  await page.goto("/login");
  await expect(page.getByRole("button", { name: /sign in/i })).toBeVisible();
  await expect(page.getByLabel(/username/i)).toBeVisible();
  await expect(page.getByLabel(/password/i)).toBeVisible();
  await page.waitForTimeout(150);
  await page.screenshot(SHOT("12-login"));
});
