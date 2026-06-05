// Apply the stored/system theme before first paint to prevent a flash of the
// wrong theme. Served from our own origin (public/ -> static root) so the
// strict CSP (`script-src 'self'`) allows it without an inline-script hash.
(function () {
  var stored = localStorage.getItem("fastaiagent-theme");
  var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  var theme =
    stored === "dark" ||
    (stored === "system" && prefersDark) ||
    (!stored && prefersDark)
      ? "dark"
      : "light";
  document.documentElement.classList.add(theme);
})();
