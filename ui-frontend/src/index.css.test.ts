import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

/**
 * Guards the skin-token invariant in index.css.
 *
 * Two complete looks live in that file, selected by a class on <html>:
 * the default "Trace Studio" (`:root` / `.dark`) and "Clean Lab"
 * (`html.skin-classic` / `html.skin-classic.dark`).
 *
 * Because `html.skin-classic` (0,1,1) out-specifies `.dark` (0,1,0), the
 * classic *dark* block must re-declare every colour token — not a delta.
 * Miss one and a classic light value silently leaks into classic dark mode,
 * which is invisible in unit tests (vitest runs with `css: false`) and easy
 * to miss by eye. These three assertions make that structural, not visual.
 */

// Resolved from the vitest root (ui-frontend/) rather than import.meta.url,
// which is not a file: URL under the jsdom environment.
const CSS = readFileSync(resolve(process.cwd(), "src/index.css"), "utf8");

/** Custom-property names declared directly in the given top-level block. */
function tokensIn(selector: string): string[] {
  // Match `selector {` at the start of a line, then take everything up to the
  // first line that closes the block at column 0. The token blocks are all
  // top-level and brace-free apart from their own wrapper.
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`^${escaped}\\s*\\{([\\s\\S]*?)^\\}`, "m");
  const body = CSS.match(re)?.[1];
  if (body === undefined) {
    throw new Error(`Block not found in index.css: ${selector}`);
  }
  return [...body.matchAll(/^\s*(--[\w-]+)\s*:/gm)].map((m) => m[1]).sort();
}

const ROOT = tokensIn(":root");
const DARK = tokensIn(".dark");
const CLASSIC = tokensIn("html.skin-classic");
const CLASSIC_DARK = tokensIn("html.skin-classic.dark");

describe("index.css skin tokens", () => {
  it("finds all four token blocks", () => {
    expect(ROOT.length).toBeGreaterThan(30);
    expect(DARK.length).toBeGreaterThan(30);
    expect(CLASSIC.length).toBeGreaterThan(30);
    expect(CLASSIC_DARK.length).toBeGreaterThan(30);
  });

  it("classic light declares exactly the same tokens as the default light", () => {
    expect(CLASSIC).toEqual(ROOT);
  });

  it("classic dark declares exactly the same tokens as the default dark", () => {
    expect(CLASSIC_DARK).toEqual(DARK);
  });

  it("every dark-overridden token exists in the light base", () => {
    // Together with the two assertions above, this is what proves the classic
    // dark block covers every colour token: the dark blocks override a subset
    // of light, and both skins override the identical subset.
    expect(ROOT).toEqual(expect.arrayContaining(DARK));
  });

  it("only --radius and the font families differ between light and dark", () => {
    const lightOnly = ROOT.filter((t) => !DARK.includes(t));
    expect(lightOnly).toEqual(["--fa-font-mono", "--fa-font-sans", "--radius"]);
  });
});
