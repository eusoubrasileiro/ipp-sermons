/**
 * Structured LLM completions via OpenRouter.
 *
 * OpenRouter is the single LLM credential for this project -- embeddings,
 * reranking and now the facet-derivation passes all go through it, so there is
 * no new secret to manage.
 *
 * Only the offline derivation scripts use this. Nothing on the request path
 * calls an LLM: search must keep working when the model is slow or down.
 */

/**
 * The workhorse for the per-sermon passes.
 *
 * Chosen for Portuguese quality and reliable structured output rather than
 * price: the whole corpus costs under two dollars to label at any current-
 * generation rate, so optimising the model down is optimising the wrong thing.
 * `google/gemini-2.5-flash-lite` is the drop-in fallback at a third the price.
 */
export const LLM_MODEL = "google/gemini-3.1-flash-lite";

/**
 * For the one-shot passes whose output is committed as ground truth -- the
 * series taxonomy and the topic taxonomy. Both are a single call over a small
 * input, so the better model costs cents and is reviewed by a human once.
 */
export const LLM_MODEL_STRONG = "google/gemini-2.5-pro";

import { postJson, RetryableError } from "./openrouter.ts";

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

export type CompletionRequest = {
  user: string;
  system?: string;
  /** JSON Schema the reply must satisfy. Enforced upstream, not by parsing. */
  schema: unknown;
  schemaName: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
};

export type LlmClient = {
  complete<T>(request: CompletionRequest): Promise<T>;
};

export type LlmOptions = {
  apiKey: string;
  model?: string;
  maxRetries?: number;
  timeoutMs?: number;
  /** Overridable so tests do not sit through the backoff. */
  retryDelayMs?: number;
  fetchImpl?: typeof fetch;
};

/**
 * Pulls the JSON object out of a reply.
 *
 * Structured output is requested with `strict: true`, but a cheap model still
 * occasionally wraps the object in a code fence or a sentence of Portuguese
 * politeness. Recovering from that is cheaper than a retry.
 */
function extractJson(content: string): unknown {
  const fenced = content.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidate = (fenced?.[1] ?? content).trim();
  try {
    return JSON.parse(candidate);
  } catch {
    const start = candidate.indexOf("{");
    const end = candidate.lastIndexOf("}");
    if (start === -1 || end <= start) throw new Error("reply contained no JSON object");
    return JSON.parse(candidate.slice(start, end + 1));
  }
}

export function createOpenRouterLlm(opts: LlmOptions): LlmClient {
  const {
    apiKey,
    model: defaultModel = LLM_MODEL,
    maxRetries = 4,
    timeoutMs = 120_000,
    retryDelayMs,
    fetchImpl,
  } = opts;

  return {
    async complete<T>(request: CompletionRequest): Promise<T> {
      const messages = [
        ...(request.system ? [{ role: "system", content: request.system }] : []),
        { role: "user", content: request.user },
      ];

      return postJson(
        {
          url: OPENROUTER_URL,
          apiKey,
          label: "completion",
          maxRetries,
          timeoutMs,
          ...(retryDelayMs === undefined ? {} : { retryDelayMs }),
          ...(fetchImpl === undefined ? {} : { fetchImpl }),
          body: {
            model: request.model ?? defaultModel,
            messages,
            temperature: request.temperature ?? 0,
            ...(request.maxTokens ? { max_tokens: request.maxTokens } : {}),
            response_format: {
              type: "json_schema",
              json_schema: { name: request.schemaName, strict: true, schema: request.schema },
            },
          },
        },
        (payload) => {
          const decoded = payload as {
            choices?: { message?: { content?: string } }[];
            error?: { message?: string };
          };

          if (decoded.error) {
            throw new RetryableError(`OpenRouter error: ${decoded.error.message ?? "unknown"}`);
          }

          const content = decoded.choices?.[0]?.message?.content;
          if (!content) throw new RetryableError("OpenRouter returned no content");

          // A non-JSON reply is worth one more attempt: it is usually the model
          // padding the object with a sentence, not a broken request.
          return extractJson(content) as T;
        },
      );
    },
  };
}
