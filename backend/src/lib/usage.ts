/**
 * What a paid run actually cost.
 *
 * OpenRouter returns `usage.cost` in dollars alongside the token counts when the
 * request asks for it, and every script here used to throw that away. The only
 * answer to "what did this corpus update cost" was arithmetic done afterwards
 * from a price list, over token counts themselves guessed from word counts --
 * two estimates stacked on each other, and no way to notice a model whose price
 * changed.
 *
 * `cachedTokens` is the number worth watching. The 56-topic catalogue rides in
 * the system prompt of all ~500 labelling calls; whether the provider is caching
 * it is the difference between paying for it once and paying for it 500 times,
 * and it is invisible without this.
 */
export type Usage = {
  promptTokens: number;
  completionTokens: number;
  cachedTokens: number;
  /** Dollars, as reported upstream — not derived from a price table here. */
  cost: number;
  calls: number;
};

const num = (v: unknown): number => (typeof v === "number" && Number.isFinite(v) ? v : 0);

/** The usage block of an OpenRouter reply, or null when it carries none. */
export function readUsage(payload: unknown): Usage | null {
  if (typeof payload !== "object" || payload === null) return null;
  const usage = (payload as { usage?: unknown }).usage;
  if (typeof usage !== "object" || usage === null) return null;

  const u = usage as {
    prompt_tokens?: unknown;
    completion_tokens?: unknown;
    cost?: unknown;
    prompt_tokens_details?: { cached_tokens?: unknown };
  };

  return {
    promptTokens: num(u.prompt_tokens),
    completionTokens: num(u.completion_tokens),
    cachedTokens: num(u.prompt_tokens_details?.cached_tokens),
    // A provider that reports tokens but not cost is normal; reporting zero is
    // honest, and inventing a price from a stale table is not.
    cost: num(u.cost),
    calls: 1,
  };
}

type UsageMeter = {
  record(usage: Usage | null): void;
  total(): Usage;
  /** One line for the end of a run, or "" when nothing was spent. */
  summary(): string;
};

export function createUsageMeter(): UsageMeter {
  const total: Usage = {
    promptTokens: 0,
    completionTokens: 0,
    cachedTokens: 0,
    cost: 0,
    calls: 0,
  };

  return {
    record(usage) {
      if (!usage) return;
      total.promptTokens += usage.promptTokens;
      total.completionTokens += usage.completionTokens;
      total.cachedTokens += usage.cachedTokens;
      total.cost += usage.cost;
      total.calls += usage.calls;
    },

    total: () => ({ ...total }),

    summary() {
      if (total.calls === 0) return "";

      const k = (n: number) => `${(n / 1000).toFixed(1)}k`;
      const cached =
        total.promptTokens > 0
          ? `, ${Math.round((100 * total.cachedTokens) / total.promptTokens)}% cached`
          : "";

      return (
        `${total.calls} call${total.calls === 1 ? "" : "s"}, ` +
        `${k(total.promptTokens)} in${cached}, ${k(total.completionTokens)} out, ` +
        `$${total.cost.toFixed(4)}`
      );
    },
  };
}
