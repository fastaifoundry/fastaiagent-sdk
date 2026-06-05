import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// Self-hosted fonts (served from our own origin so the strict CSP needs no
// external font-CDN allowance). Weights match the prior Google Fonts request:
// DM Sans 400/500/600/700, JetBrains Mono 400/500/600. Family names ('DM Sans',
// 'JetBrains Mono') match the --font-sans / --font-mono vars in index.css.
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
