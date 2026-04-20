/**
 * Lightweight logger utility for FastAIAgent frontend.
 *
 * - In development: all log levels are active.
 * - In production: only errors are logged.
 *
 * Usage:
 *   import { logger } from "@/lib/logger";
 *   logger.debug("Subscription fetched", data, "Metering");
 *   logger.error("Stream failed", err, "ChatStore");
 *   logger.apiError({ method: "POST", path: "/execute", status: 402 });
 */

type LogLevel = "debug" | "info" | "warn" | "error";

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

// In production, only errors. In development, everything.
const CURRENT_LEVEL: number = import.meta.env.PROD
  ? LOG_LEVELS.error
  : LOG_LEVELS.debug;

function shouldLog(level: LogLevel): boolean {
  return LOG_LEVELS[level] >= CURRENT_LEVEL;
}

interface ApiErrorContext {
  method?: string;
  path?: string;
  status?: number;
  detail?: unknown;
  duration?: number;
}

function formatPrefix(level: LogLevel, tag?: string): string {
  const timestamp = new Date().toISOString();
  const tagStr = tag ? ` [${tag}]` : "";
  return `[${timestamp}] [${level.toUpperCase()}]${tagStr}`;
}

export const logger = {
  debug(message: string, data?: unknown, tag?: string): void {
    if (!shouldLog("debug")) return;
    if (data !== undefined) {
      console.debug(formatPrefix("debug", tag), message, data);
    } else {
      console.debug(formatPrefix("debug", tag), message);
    }
  },

  info(message: string, data?: unknown, tag?: string): void {
    if (!shouldLog("info")) return;
    if (data !== undefined) {
      console.info(formatPrefix("info", tag), message, data);
    } else {
      console.info(formatPrefix("info", tag), message);
    }
  },

  warn(message: string, data?: unknown, tag?: string): void {
    if (!shouldLog("warn")) return;
    if (data !== undefined) {
      console.warn(formatPrefix("warn", tag), message, data);
    } else {
      console.warn(formatPrefix("warn", tag), message);
    }
  },

  error(message: string, error?: unknown, tag?: string): void {
    if (!shouldLog("error")) return;
    if (error !== undefined) {
      console.error(formatPrefix("error", tag), message, error);
    } else {
      console.error(formatPrefix("error", tag), message);
    }
  },

  /** Structured API error log — always logged (errors always pass threshold). */
  apiError(context: ApiErrorContext, error?: unknown): void {
    if (!shouldLog("error")) return;
    console.error(
      formatPrefix("error", "API"),
      `${context.method || "?"} ${context.path || "?"} → ${context.status || "?"}`,
      {
        ...context,
        error: error instanceof Error ? error.message : error,
      },
    );
  },

  /** Structured API request log — debug level, for tracing request flow. */
  apiRequest(method: string, path: string): void {
    if (!shouldLog("debug")) return;
    console.debug(formatPrefix("debug", "API"), `${method} ${path}`);
  },
};
