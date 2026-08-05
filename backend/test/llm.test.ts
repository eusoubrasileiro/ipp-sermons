import { describe, expect, it, vi } from "vitest";
import {
  type CompletionRequest,
  createOpenRouterLlm,
  LLM_MODEL,
  type LlmClient,
  type LlmOptions,
} from "../src/lib/llm.ts";
import { FatalOpenRouterError } from "../src/lib/openrouter.ts";

const SCHEMA = {
  type: "object",
  properties: { livro: { type: ["string", "null"] } },
  required: ["livro"],
  additionalProperties: false,
} as const;

const reply = (content: string) =>
  new Response(JSON.stringify({ choices: [{ message: { content } }] }), { status: 200 });

const REQUEST: CompletionRequest = { user: "Efésios 5", schema: SCHEMA, schemaName: "ref" };

const ask = (fetchImpl: typeof fetch, overrides: Partial<LlmOptions> = {}) => {
  const client: LlmClient = createOpenRouterLlm({
    apiKey: "k",
    fetchImpl,
    retryDelayMs: 0,
    ...overrides,
  });
  return client.complete(REQUEST);
};

describe("createOpenRouterLlm", () => {
  it("returns the parsed object, not the raw text", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(reply('{"livro":"efesios"}'));
    await expect(ask(fetchImpl as unknown as typeof fetch)).resolves.toEqual({ livro: "efesios" });
  });

  it("asks for a strict json schema, so the model cannot free-form", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(reply('{"livro":null}'));
    await ask(fetchImpl as unknown as typeof fetch);

    const [, init] = fetchImpl.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.model).toBe(LLM_MODEL);
    expect(body.response_format.type).toBe("json_schema");
    expect(body.response_format.json_schema.strict).toBe(true);
    expect(body.response_format.json_schema.schema).toEqual(SCHEMA);
  });

  it("sends the system prompt when there is one", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(reply('{"livro":null}'));
    await createOpenRouterLlm({
      apiKey: "k",
      fetchImpl: fetchImpl as unknown as typeof fetch,
    }).complete({ system: "responda em json", user: "x", schema: SCHEMA, schemaName: "ref" });

    const [, init] = fetchImpl.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.messages).toEqual([
      { role: "system", content: "responda em json" },
      { role: "user", content: "x" },
    ]);
  });

  it("retries a rate limit and then succeeds", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(new Response("slow down", { status: 429 }))
      .mockResolvedValue(reply('{"livro":"atos"}'));

    await expect(ask(fetchImpl as unknown as typeof fetch)).resolves.toEqual({ livro: "atos" });
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it("retries an upstream 5xx", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(new Response("bad gateway", { status: 502 }))
      .mockResolvedValue(reply('{"livro":"atos"}'));

    await expect(ask(fetchImpl as unknown as typeof fetch)).resolves.toEqual({ livro: "atos" });
  });

  it("does not retry a request bug, because every attempt is billed", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(new Response("bad schema", { status: 400 }));

    await expect(ask(fetchImpl as unknown as typeof fetch)).rejects.toThrow(FatalOpenRouterError);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("retries when the model returns text that is not json", async () => {
    // A cheap model occasionally wraps its answer in prose or a code fence.
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(reply("Claro! Aqui está: {...}"))
      .mockResolvedValue(reply('{"livro":"joao"}'));

    await expect(ask(fetchImpl as unknown as typeof fetch)).resolves.toEqual({ livro: "joao" });
  });

  it("unwraps a fenced code block rather than failing on it", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(reply('```json\n{"livro":"rute"}\n```'));
    await expect(ask(fetchImpl as unknown as typeof fetch)).resolves.toEqual({ livro: "rute" });
  });

  it("gives up after the retry budget and says why", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(new Response("nope", { status: 500 }));

    await expect(ask(fetchImpl as unknown as typeof fetch, { maxRetries: 2 })).rejects.toThrow(
      /after 2 retries/,
    );
    expect(fetchImpl).toHaveBeenCalledTimes(3);
  });

  it("surfaces an error object returned with a 200", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ error: { message: "no credits" } }), { status: 200 }),
      );

    await expect(ask(fetchImpl as unknown as typeof fetch, { maxRetries: 0 })).rejects.toThrow(
      /no credits/,
    );
  });

  it("aborts a hanging request instead of waiting forever", async () => {
    const fetchImpl = vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
        }),
    );

    await expect(
      ask(fetchImpl as unknown as typeof fetch, { maxRetries: 0, timeoutMs: 20 }),
    ).rejects.toThrow(/aborted|failed/);
  });
});
