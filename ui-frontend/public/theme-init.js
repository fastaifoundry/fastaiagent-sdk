// Apply the stored/system theme and the stored skin before first paint, to
// prevent a flash of the wrong colours. Served from our own origin
// (public/ -> static root) so the strict CSP (`script-src 'self'`) allows it
// without an inline-script hash.
//
// The default skin ("Trace Studio") owns the base token blocks in index.css,
// so the common path adds no class at all — only an explicit opt-in to the
// classic look needs one.
(function () {
  // localStorage throws in hardened/private contexts; a throw here would
  // otherwise leave the page entirely unthemed.
  var stored = null;
  var skin = null;
  try {
    stored = localStorage.getItem("fastaiagent-theme");
    skin = localStorage.getItem("fastaiagent-ui-skin");
  } catch (e) {
    /* no persisted preference available — fall back to system/defaults */
  }

  var prefersDark = false;
  try {
    prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  } catch (e) {
    /* matchMedia unavailable — assume light */
  }

  var theme =
    stored === "dark" ||
    (stored === "system" && prefersDark) ||
    (!stored && prefersDark)
      ? "dark"
      : "light";
  document.documentElement.classList.add(theme);

  if (skin === "classic") {
    document.documentElement.classList.add("skin-classic");
  }
})();
