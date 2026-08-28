/**
 * The prompt row must be clickable across its whole width.
 *
 * Regression: `stopPropagation` was attached to the name *cell* rather than
 * the anchor inside it, so every click landing in the cell's blank area was
 * swallowed. The name column is by far the widest, which left ~860px of each
 * row showing a pointer cursor and doing nothing, while the narrow numeric
 * cells on the right worked fine.
 *
 * The unit test that covered "click anywhere in the row" clicked a cell that
 * *had text*, so it never touched the dead zone. This drives a real browser
 * with real layout, which is the only place the gap exists.
 */
import { expect, test } from "@playwright/test";

for (const skin of ["next", "classic"]) {
  test(`name-cell dead zone is clickable — ${skin}`, async ({ page }) => {
    await page.addInitScript((s) => {
      window.localStorage.setItem("fastaiagent-ui-skin", s as string);
    }, skin);
    await page.goto("/prompts");
    await expect(page.getByRole("heading", { name: /^Prompts$/i })).toBeVisible();

    const row = page.getByRole("row").filter({ hasText: "support-greeting" }).first();
    const cell = row.getByRole("cell").nth(0);
    const box = (await cell.boundingBox())!;
    const link = (await row.getByRole("link").first().boundingBox())!;
    console.log(`  [${skin}] gap = ${Math.round(box.width - link.width)}px`);

    // Far right of the name cell — deep in the former dead zone.
    await cell.click({ position: { x: box.width - 20, y: box.height / 2 } });
    await page.waitForTimeout(500);
    console.log(`  [${skin}] gap click -> ${new URL(page.url()).pathname}`);
    expect(page.url()).toContain("/prompts/support-greeting");

    // The name itself must still work, and still navigate exactly once.
    await page.goto("/prompts");
    await row.getByRole("link").first().click();
    await page.waitForTimeout(500);
    console.log(`  [${skin}] name click -> ${new URL(page.url()).pathname}`);
    expect(page.url()).toContain("/prompts/support-greeting");
  });
}

test("dataset rows are clickable across their whole width too", async ({
  page,
}) => {
  // The Datasets list had the identical cell-level stopPropagation, so the
  // same ~600px dead zone. Fixed alongside Prompts; guarded here so the
  // pattern can't creep back into either list.
  await page.goto("/datasets");
  await page.waitForTimeout(500);

  const row = page.getByRole("row").filter({ hasText: /\w/ }).nth(1);
  const cell = row.getByRole("cell").nth(0);
  const box = await cell.boundingBox();
  const link = row.getByRole("link").first();
  if (!box || (await link.count()) === 0) {
    test.skip(true, "no datasets seeded on this server");
    return;
  }
  await cell.click({ position: { x: box.width - 20, y: box.height / 2 } });
  await page.waitForTimeout(500);
  expect(page.url()).toContain("/datasets/");
});
