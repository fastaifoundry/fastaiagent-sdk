/** The server's bounds (`PlaygroundParameters.max_tokens`: ge=1, le=200_000). */
export const MAX_TOKENS_MIN = 1;
export const MAX_TOKENS_MAX = 200_000;

/** True when `n` is a max_tokens the server will accept (1.89.0, P1). */
export function isValidMaxTokens(n: number): boolean {
  return Number.isInteger(n) && n >= MAX_TOKENS_MIN && n <= MAX_TOKENS_MAX;
}
