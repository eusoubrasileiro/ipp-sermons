/**
 * The OpenRouter request loop shared by embeddings, reranking and the facet
 * derivation passes.
 *
 * All three want the same thing and learned it the same expensive way: a
 * per-attempt deadline, exponential backoff on rate limits and upstream 5xx,
 * and an immediate stop on anything a retry cannot fix. The timeout is the part
 * that matters most -- a full corpus run once parked forever on a dropped
 * connection because fetch had no deadline, so the retry loop never ran.
 *
 * Callers supply a `parse` that turns the decoded body into their own shape.
 * It can ask for another attempt by throwing `RetryableError` (a truncated or
 * short batch) or abandon the call by throwing `FatalOpenRouterError` (a
 * response that violates a contract no retry will satisfy).
 */
export class FatalOpenRouterError extends Error {
  // Typed as string rather than a literal so subclasses can name themselves --
  // `FatalEmbeddingError` narrows the contract further for the pgvector case.
  override readonly name: string = "FatalOpenRouterError";
}

/** Thrown by a `parse` that wants the request attempted again. */
export class RetryableError extends Error {
  override readonly name = "RetryableError";
}

export type OpenRouterCall = {
  url: string;
  apiKey: string;
  body: unknown;
  /** Names the operation in the give-up message: "embedding", "completion". */
  label: string;
  maxRetries?: number;
  timeoutMs?: number;
  /** Overridable so tests do not sit through the backoff. */
  retryDelayMs?: number;
  fetchImpl?: typeof fetch;
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function postJson<T>(
  call: OpenRouterCall,
  parse: (payload: unknown) => T,
): Promise<T> {
  const {
    url,
    apiKey,
    body,
    label,
    maxRetries = 5,
    timeoutMs = 60_000,
    retryDelayMs = 500,
    fetchImpl = fetch,
  } = call;

  const payload = JSON.stringify(body);
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    if (attempt > 0) await sleep(Math.min(2 ** attempt * retryDelayMs, 16_000));

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const res = await fetchImpl(url, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: payload,
        signal: controller.signal,
      });

      if (res.status === 429 || res.status >= 500) {
        lastError = new Error(`OpenRouter ${res.status}: ${await res.text()}`);
        continue;
      }

      // 4xx other than rate limiting is a request bug -- retrying wastes money
      // and time, so surface it immediately.
      if (!res.ok) throw new FatalOpenRouterError(`OpenRouter ${res.status}: ${await res.text()}`);

      return parse(await res.json());
    } catch (err) {
      if (err instanceof FatalOpenRouterError) throw err;
      lastError = err instanceof Error ? err : new Error(String(err));
    } finally {
      clearTimeout(timer);
    }
  }

  throw new Error(`${label} failed after ${maxRetries} retries: ${lastError?.message}`);
}
