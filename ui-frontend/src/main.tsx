import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// Self-hosted fonts (served from our own origin so the strict CSP needs no
// external font-CDN allowance). Both skins ship their families, selected at
// runtime by --fa-font-sans / --fa-font-mono in index.css.
//
// Default skin — IBM Plex Sans 400/500/600/700 + IBM Plex Mono 400/500/600.
// Imported per subset (latin, latin-ext) rather than via the default
// entrypoints, which would also pull cyrillic/cyrillic-ext/greek/vietnamese —
// ~450 KiB of files this UI never renders.
import "@fontsource/ibm-plex-sans/latin-400.css";
import "@fontsource/ibm-plex-sans/latin-ext-400.css";
import "@fontsource/ibm-plex-sans/latin-500.css";
import "@fontsource/ibm-plex-sans/latin-ext-500.css";
import "@fontsource/ibm-plex-sans/latin-600.css";
import "@fontsource/ibm-plex-sans/latin-ext-600.css";
import "@fontsource/ibm-plex-sans/latin-700.css";
import "@fontsource/ibm-plex-sans/latin-ext-700.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-ext-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "@fontsource/ibm-plex-mono/latin-ext-500.css";
import "@fontsource/ibm-plex-mono/latin-600.css";
import "@fontsource/ibm-plex-mono/latin-ext-600.css";

// Classic skin — DM Sans 400/500/600/700, JetBrains Mono 400/500/600.
import "@fontsource/dm-sans/400.css";
import "@fontsource/dm-sans/500.css";
import "@fontsource/dm-sans/600.css";
import "@fontsource/dm-sans/700.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "@fontsource/jetbrains-mono/600.css";
import "./index.css";
import App from "./App";

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("#root not found — is index.html correct?");
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>
);
