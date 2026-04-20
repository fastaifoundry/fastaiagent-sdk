import { useTheme } from "@/components/theme/ThemeProvider";
import logoDark from "@/assets/logos/fastaiagent-dark.svg";
import logoLight from "@/assets/logos/fastaiagent-light.svg";
import faviconDark from "@/assets/logos/fastaiagent-favicon-dark.svg";
import faviconLight from "@/assets/logos/fastaiagent-favicon-light.svg";

interface FastAIAgentLogoProps {
  className?: string;
  variant?: "full" | "favicon";
}

export function FastAIAgentLogo({ className = "h-8 w-8", variant = "full" }: FastAIAgentLogoProps) {
  const { resolvedTheme } = useTheme();

  // Light SVG on dark background, dark SVG on light background
  const logo =
    variant === "favicon"
      ? resolvedTheme === "dark" ? faviconLight : faviconDark
      : resolvedTheme === "dark" ? logoLight : logoDark;

  return (
    <div className={`rounded-md border border-border p-0.5 ${className}`}>
      <img src={logo} alt="FastAIAgent" className="h-full w-full" />
    </div>
  );
}
