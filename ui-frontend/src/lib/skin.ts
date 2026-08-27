/**
 * UI skin selection.
 *
 * Two complete looks ship in the Local UI, selected by a class on <html>:
 *
 *   "next"    — the default "Trace Studio" look, matching the Enterprise
 *               console: icon rail + flyout nav, indigo palette, IBM Plex.
 *   "classic" — the original Local UI: fixed sidebar, "Clean Lab" palette,
 *               DM Sans / JetBrains Mono.
 *
 * The default owns the base token blocks in index.css, so only "classic"
 * carries a class (`skin-classic`). See the SKINS comment in index.css.
 *
 * The pre-paint application lives in public/theme-init.js — this module is
 * only for reading the current value and switching it.
 */

export type Skin = "next" | "classic";

const STORAGE_KEY = "fastaiagent-ui-skin";
const CLASSIC_CLASS = "skin-classic";

export function getSkin(): Skin {
  try {
    return localStorage.getItem(STORAGE_KEY) === "classic" ? "classic" : "next";
  } catch {
    // Hardened/private contexts deny localStorage; fall back to the default.
    return "next";
  }
}

/**
 * Persist the skin and reload.
 *
 * The reload is deliberate rather than a React state flip: switching skins
 * swaps the layout element, which remounts the entire page subtree anyway, so
 * there is no in-flight state worth preserving. Reloading also exercises the
 * pre-paint path in theme-init.js and the font stack exactly as a cold load
 * does — which is the state we actually want to be exercising.
 */
export function setSkin(skin: Skin): void {
  try {
    localStorage.setItem(STORAGE_KEY, skin);
  } catch {
    // Can't persist — still apply for this session so the toggle does something.
    document.documentElement.classList.toggle(
      CLASSIC_CLASS,
      skin === "classic"
    );
    return;
  }
  window.location.reload();
}
