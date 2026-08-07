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
 * The workhorse for the per-sermon passes: scripture extraction and topic
 * labelling, ~460 calls each.
 *
 * Both are mechanical work against a strict schema -- read what the preacher
 * announced, or pick from a closed list -- so the job is reliable structured
 * output at volume, not reasoning. This is the cheapest current-generation
 * model that offers a 1M context and strict `json_schema`.
 *
 * `google/gemini-3.5-flash-lite` is the drop-in fallback if Portuguese output
 * disappoints. Do NOT reach for the DeepSeek flash models here, cheap as they
 * are: they are the one family with documented language drift
 * (deepseek-ai/DeepSeek-V3 issues #1045, #1255, #1463), and this corpus is
 * entirely Portuguese.
 */
export const LLM_MODEL = "openai/gpt-5.6-luna";

/**
 * For the one-shot passes whose output is committed as ground truth -- the
 * series taxonomy and the topic taxonomy. One call over a small input, so the
 * better model costs cents and a human reviews the result once.
 *
 * `qwen/qwen3.7-plus` was the first choice on evidence -- it is the only family
 * with *measured* Brazilian-Portuguese strength rather than inferred, ranking
 * 4th of 16 on Prosa, a benchmark of 1,000 real pt-BR conversations (arXiv
 * 2605.01630), ahead of Gemini 3 Flash and GPT-4.1, and ahead of Brazil's own
 * Sabia-4 on conversational Portuguese by 12 points (arXiv 2603.10213).
 *
 * It is not used, because it could not complete this call: every attempt failed
 * through OpenRouter and the script died after the full retry budget. A model
 * that will not answer is worth nothing regardless of its benchmark. Revisit if
 * its availability improves.
 */
export const LLM_MODEL_STRONG = "openai/gpt-5.6-luna";

import { postJson, RetryableError } from "./openrouter.ts";
import type { Usage } from "./usage.ts";

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
  /** Where a run adds up what it spent — see lib/usage.ts. */
  onUsage?: (usage: Usage | null) => void;
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
    onUsage,
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
          ...(onUsage === undefined ? {} : { onUsage }),
          body: {
            model: request.model ?? defaultModel,
            messages,
            temperature: request.temperature ?? 0,
            // Without this the reply carries token counts but no `cost`, and the
            // only way back to dollars is a price table that goes stale.
            usage: { include: true },
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
