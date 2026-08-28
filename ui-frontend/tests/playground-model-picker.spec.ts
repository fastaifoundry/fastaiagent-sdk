/**
 * Regression tests for the Playground's // MODEL card (1.53.0).
 *
 * Covers the three things that were broken or missing at 1.52.0:
 *
 *  1. Switching provider left the previous provider's model selected, so the
 *     next run posted a mismatched pair (e.g. anthropic + gpt-4o-mini) that
 *     the provider rejects with a 404.
 *  2. The model picker was a fixed <Select>, so a model the shipped catalog
 *     didn't list was unreachable even though LLMClient accepts any id.
 *  3. temperature / top_p were always sent, which Anthropic rejects. They are
 *     now opt-in and read "auto" until switched on.
 *
 * Every assertion runs against **both** UI skins. The two skins share one
 * route and one PlaygroundPage component — only the shell and CSS tokens
 * differ — but classic re-declares every colour token, so the controls are
 * checked in both rather than assumed.
 *
 * No API key required: nothing here issues an LLM call.
 */
import { expect, test, type Page } from "@playwright/test";

const SKINS = ["next", "classic"] as const;

/** Pin the skin before first paint, the same way theme-init.js does. */
async function gotoPlayground(page: Page, skin: (typeof SKINS)[number]) {
  await page.addInitScript((value) => {
    window.localStorage.setItem("fastaiagent-ui-skin", value);
  }, skin);
  await page.goto("/playground");
  await expect(
    page.getByRole("heading", { name: /^Playground$/i }),
  ).toBeVisible();
}

const modelInput = (page: Page) => page.locator("#playground-model");

for (const skin of SKINS) {
  test.describe(`playground model picker — ${skin} skin`, () => {
    test("switching provider repoints the model", async ({ page }) => {
      await gotoPlayground(page, skin);

      const before = await modelInput(page).inputValue();
      expect(before.length).toBeGreaterThan(0);

      // Open the provider select and pick a different, key-bearing provider.
      const trigger = page.locator("#playground-provider");
      const providerBefore = (await trigger.textContent())?.trim();
      await trigger.click();
      const target = page
        .getByRole("option")
        .filter({ hasNotText: "(no key)" })
        .filter({ hasNotText: providerBefore ?? "" })
        .first();
      await target.click();

      // The model must be repointed at the new provider's first entry, never
      // left pointing at the old provider's model.
      await expect(modelInput(page)).not.toHaveValue(before);
      expect((await modelInput(page).inputValue()).length).toBeGreaterThan(0);
    });

    test("model field accepts a model that is not in the suggestions", async ({
      page,
    }) => {
      await gotoPlayground(page, skin);

      const input = modelInput(page);
      await input.fill("some-unreleased-model-2099");
      await expect(input).toHaveValue("some-unreleased-model-2099");

      // Suggestions still exist alongside the free text.
      const options = page.locator("#playground-model-options option");
      expect(await options.count()).toBeGreaterThan(0);
    });

    test("Run is disabled when the model is cleared", async ({ page }) => {
      await gotoPlayground(page, skin);

      await page.locator("#playground-template").fill("Say pong.");
      const run = page.getByRole("button", { name: /^Run$/ });

      await modelInput(page).fill("");
      await expect(run).toBeDisabled();

      await modelInput(page).fill("gpt-4o-mini");
      await expect(run).toBeEnabled();
    });

    test("temperature and top_p default to auto and are opt-in", async ({
      page,
    }) => {
      await gotoPlayground(page, skin);

      await page.getByRole("button", { name: /Parameters/ }).click();

      // Both read "auto" until explicitly enabled — that is what keeps them
      // out of the request body and keeps Anthropic working.
      const auto = page.getByText("auto", { exact: true });
      await expect(auto.first()).toBeVisible();
      await expect(auto).toHaveCount(2);

      const tempSwitch = page.getByRole("switch", { name: "Send temperature" });
      const topPSwitch = page.getByRole("switch", { name: "Send top_p" });
      await expect(tempSwitch).not.toBeChecked();
      await expect(topPSwitch).not.toBeChecked();

      // Enabling temperature swaps its "auto" for a concrete value.
      await tempSwitch.click();
      await expect(tempSwitch).toBeChecked();
      await expect(auto).toHaveCount(1);
      await expect(page.getByText("1.00", { exact: true }).first()).toBeVisible();
    });
  });
}
