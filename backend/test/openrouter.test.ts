import { describe, expect, it, vi } from "vitest";
import {
  FatalOpenRouterError,
  type OpenRouterCall,
  postJson,
  RetryableError,
} from "../src/lib/openrouter.ts";

const ok = (payload: unknown) => new Response(JSON.stringify(payload), { status: 200 });

const call = (
  fetchImpl: typeof fetch,
  overrides: Partial<OpenRouterCall> = {},
): OpenRouterCall => ({
  url: "https://openrouter.ai/api/v1/test",
  apiKey: "k",
  body: { model: "m" },
  label: "test call",
  retryDelayMs: 0,
  fetchImpl,
  ...overrides,
});

describe("postJson", () => {
  it("sends the credential and the body as JSON", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(ok({ value: 1 }));
    await postJson(call(fetchImpl as unknown as typeof fetch), (p) => p);

    const [url, init] = fetchImpl.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://openrouter.ai/api/v1/test");
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer k");
    expect(init.body).toBe('{"model":"m"}');
  });

  it("hands the decoded payload to the parser", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(ok({ value: 7 }));
    const out = await postJson(
      call(fetchImpl as unknown as typeof fetch),
      (p) => (p as { value: number }).value * 2,
    );
    expect(out).toBe(14);
  });

  it("retries when the parser asks for another attempt", async () => {
    // A short batch or a truncated reply is worth one more try; the caller
    // signals that with RetryableError rather than by returning a sentinel.
    // A fresh Response per call, because a body can only be read once.
    const fetchImpl = vi.fn(async () => ok({ value: 1 }));
    let seen = 0;

    const out = await postJson(call(fetchImpl as unknown as typeof fetch), () => {
      seen++;
      if (seen < 3) throw new RetryableError("short batch");
      return "done";
    });

    expect(out).toBe("done");
    expect(fetchImpl).toHaveBeenCalledTimes(3);
  });

  it("stops immediately when the parser declares the contract broken", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(ok({ value: 1 }));

    await expect(
      postJson(call(fetchImpl as unknown as typeof fetch), () => {
        throw new FatalOpenRouterError("wrong dimensions");
      }),
    ).rejects.toThrow(/wrong dimensions/);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("retries a rate limit and an upstream 5xx", async () => {
    let call_ = 0;
    const fetchImpl = vi.fn(async () => {
      call_++;
      if (call_ === 1) return new Response("slow", { status: 429 });
      if (call_ === 2) return new Response("boom", { status: 503 });
      return ok({ value: 1 });
    });

    await expect(postJson(call(fetchImpl as unknown as typeof fetch), () => "ok")).resolves.toBe(
      "ok",
    );
    expect(fetchImpl).toHaveBeenCalledTimes(3);
  });

  it("does not retry a request bug, because every attempt is billed", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(new Response("bad model", { status: 400 }));

    await expect(postJson(call(fetchImpl as unknown as typeof fetch), () => "ok")).rejects.toThrow(
      FatalOpenRouterError,
    );
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("names the operation when it gives up", async () => {
    const fetchImpl = vi.fn(async () => new Response("nope", { status: 500 }));

    await expect(
      postJson(call(fetchImpl as unknown as typeof fetch, { maxRetries: 1 }), () => "ok"),
    ).rejects.toThrow(/test call failed after 1 retries: OpenRouter 500/);
  });

  it("aborts a hanging request instead of waiting forever", async () => {
    // The stall that motivated the deadline: fetch that never settles, so the
    // retry loop never even runs.
    const fetchImpl = vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
        }),
    );

    await expect(
      postJson(
        call(fetchImpl as unknown as typeof fetch, { maxRetries: 1, timeoutMs: 20 }),
        () => "ok",
      ),
    ).rejects.toThrow(/test call failed/);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });
});
