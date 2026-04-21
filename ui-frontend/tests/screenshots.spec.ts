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
  await expect(page.getByText("agent.support-bot").first()).toBeVisible();
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

test("13 — Analytics", async ({ page }) => {
  await page.goto("/analytics");
  await expect(page.getByRole("heading", { name: /Analytics/i })).toBeVisible();
  await expect(page.getByText(/Latency percentiles/i)).toBeVisible();
  await page.waitForTimeout(500);
  await page.screenshot(SHOT("13-analytics"));
});

test("14 — Thread view", async ({ page }) => {
  await page.goto("/threads/session-demo");
  await expect(page.getByRole("heading", { name: /Thread/i })).toBeVisible();
  await page.waitForTimeout(250);
  await page.screenshot(SHOT("14-thread"));
});

test("15 — Trace detail with scores card", async ({ page }) => {
  await page.goto("/traces/aaaaaaaaaaaaaaaaaaaaaaaaaaaa1111");
  await expect(page.getByText(/Scores/i).first()).toBeVisible();
  await page.waitForTimeout(400);
  await page.screenshot(SHOT("15-trace-scores"));
});

test("16 — KB list", async ({ page }) => {
  await page.goto("/kb");
  await expect(page.getByRole("heading", { name: /Knowledge Bases/i })).toBeVisible();
  await expect(page.getByText("support-kb").first()).toBeVisible();
  await page.waitForTimeout(200);
  await page.screenshot(SHOT("16-kb-list"));
});

test("17 — KB detail — documents tab", async ({ page }) => {
  await page.goto("/kb/support-kb");
  await expect(page.getByRole("heading", { name: /support-kb/i })).toBeVisible();
  await expect(page.getByRole("tab", { name: /Documents/i })).toBeVisible();
  // First document preview from the seeded corpus.
  await expect(page.getByText(/Refund policy/i).first()).toBeVisible();
  await page.waitForTimeout(250);
  await page.screenshot(SHOT("17-kb-documents"));
});

test("18 — KB detail — search playground", async ({ page }) => {
  await page.goto("/kb/support-kb");
  await page.getByRole("tab", { name: /Search playground/i }).click();
  const queryInput = page.getByLabel(/Query/i);
  await queryInput.fill("refund policy");
  await page.getByRole("button", { name: /^Run$/ }).click();
  await expect(page.getByText(/result/i).first()).toBeVisible();
  await page.waitForTimeout(400);
  await page.screenshot(SHOT("18-kb-search"));
});

test("19 — KB detail — lineage tab", async ({ page }) => {
  await page.goto("/kb/support-kb");
  await page.getByRole("tab", { name: /Lineage/i }).click();
  // Seed writes 3 retrieval spans attributed to support-bot.
  await expect(page.getByText(/support-bot/i).first()).toBeVisible();
  await page.waitForTimeout(300);
  await page.screenshot(SHOT("19-kb-lineage"));
});

test("20 — Agent Replay comparison view", async ({ page }) => {
  // CI / the docs snapshot doesn't have LLM credentials, so we can't
  // actually rerun. Stub the fork + rerun + compare routes with a
  // realistic refund-bot bug fix so the ReplayDiffView renders with
  // meaningful content for the screenshot.

  const traceId = "aaaaaaaaaaaaaaaaaaaaaaaaaaaa1111"; // seeded trace
  const forkId = "fork-screenshot-0001";
  const rerunTraceId = "aaaaaaaaaaaaaaaaaaaaaaaaaaaa2222";

  const originalSteps = [
    {
      step: 0,
      span_name: "agent.support-bot",
      span_id: "s0",
      input: { query: "When do refunds get processed?" },
      output: {},
      attributes: { "agent.name": "support-bot" },
      timestamp: new Date().toISOString(),
    },
    {
      step: 1,
      span_name: "llm.chat",
      span_id: "s1",
      input: {
        system: "You are a customer support agent. Help the user.",
        user: "When do refunds get processed?",
      },
      output: { text: "Refunds are processed within 14 business days." },
      attributes: { "gen_ai.request.model": "gpt-4o-mini" },
      timestamp: new Date().toISOString(),
    },
  ];

  const newSteps = [
    originalSteps[0],
    {
      step: 1,
      span_name: "llm.chat",
      span_id: "s1-rerun",
      input: {
        system:
          "You are a customer support agent. Refunds are processed within 7 business days of receiving the return. Reply in one sentence.",
        user: "When do refunds get processed?",
      },
      output: { text: "Refunds are processed within 7 business days." },
      attributes: { "gen_ai.request.model": "gpt-4o-mini" },
      timestamp: new Date().toISOString(),
    },
  ];

  // Stub POST /api/replay/{traceId}/fork
  await page.route(`**/api/replay/${traceId}/fork`, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ fork_id: forkId }),
    })
  );

  // Stub PATCH /api/replay/forks/{forkId}
  await page.route(`**/api/replay/forks/${forkId}`, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ fork_id: forkId, applied: true }),
    })
  );

  // Stub POST /api/replay/forks/{forkId}/rerun
  await page.route(`**/api/replay/forks/${forkId}/rerun`, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        fork_id: forkId,
        new_trace_id: rerunTraceId,
        new_output: { text: "Refunds are processed within 7 business days." },
        original_output: {
          text: "Refunds are processed within 14 business days.",
        },
        steps_executed: newSteps.length,
      }),
    })
  );

  // Stub GET /api/replay/forks/{forkId}/compare
  await page.route(
    `**/api/replay/forks/${forkId}/compare*`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          original_steps: originalSteps,
          new_steps: newSteps,
          diverged_at: 1,
        }),
      })
  );

  await page.goto(`/traces/${traceId}/replay`);
  await expect(page.getByRole("heading", { name: /Agent Replay/i })).toBeVisible();

  // Pick the LLM span so Fork is enabled.
  await page.getByText("llm.chat").first().click();
  await page.getByRole("button", { name: /Fork here/i }).click();
  await expect(page.getByRole("heading", { name: /Fork and rerun/i })).toBeVisible();

  // Type a new prompt so the modify step also fires (stubbed).
  await page
    .locator("textarea")
    .first()
    .fill(
      "You are a customer support agent. Refunds are processed within 7 business days."
    );

  await page.getByRole("button", { name: /Rerun from this step/i }).click();

  // Wait for ReplayDiffView to render — give it more time than default
  // because the client runs 3 sequential requests (fork, modify, rerun)
  // before onRerunComplete fires. "Rerun complete" text appears twice
  // (card heading + sonner toast) so pick the first one.
  await expect(page.getByText(/Rerun complete/i).first()).toBeVisible({
    timeout: 15000,
  });
  await expect(page.getByText(/diverged at step 1/i)).toBeVisible({
    timeout: 10000,
  });

  // Expand the diverged step so the screenshot shows input+output diff content.
  const expandBtn = page.getByRole("button", { name: /expand step diff/i }).first();
  await expandBtn.click();
  // The DiffBlock labels "Input"/"Output" appear once each inside the expanded
  // row — match the first to dodge other instances (inspector tabs, etc).
  await expect(page.getByText("Input").first()).toBeVisible();

  await page.waitForTimeout(400);
  await page.screenshot(SHOT("20-replay-comparison"));
});
