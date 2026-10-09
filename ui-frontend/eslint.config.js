// `npm run lint`. The `lint` script shipped with the Vite template in v0.8.0
// but eslint itself was never added, so it could not run until 1.86.0.
import js from "@eslint/js";
import { defineConfig, globalIgnores } from "eslint/config";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";
import tseslint from "typescript-eslint";

export default defineConfig([
  globalIgnores(["dist", "coverage"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [js.configs.recommended, tseslint.configs.recommended],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      // The two established hook rules. The plugin's newer presets add React
      // Compiler rules, which assume code written for the compiler.
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      // Deliberate non-component exports next to their components: the theme
      // hook beside its provider, shadcn's variant helpers, and a test hook.
      "react-refresh/only-export-components": [
        "warn",
        {
          allowConstantExport: true,
          allowExportNames: [
            "useTheme",
            "badgeVariants",
            "buttonVariants",
            "_testOnlyHasMultimodal",
          ],
        },
      ],
    },
  },
]);
