import { describe, expect, it } from "vitest";
import { createUsageMeter, readUsage } from "../src/lib/usage.ts";

/**
 * What a paid run actually cost, rather than what someone estimated afterwards.
 *
 * OpenRouter returns `usage.cost` in dollars alongside the token counts when the
 * request asks for it, and every script here was throwing that away -- so the
 * only answer to "what did the corpus update cost" was arithmetic done by hand
 * from a price list, on token counts that were themselves guessed from word
 * counts. Two estimates stacked on each other.
 */
const response = (over: Record<string, unknown> = {}) => ({
  choices: [{ message: { content: "{}" } }],
  usage: {
    prompt_tokens: 3400,
    completion_tokens: 120,
    cost: 0.000412,
    prompt_tokens_details: { cached_tokens: 1500 },
    ...over,
  },
});

describe("readUsage", () => {
  it("reads tokens and dollars off a reply", () => {
    expect(readUsage(response())).toEqual({
      promptTokens: 3400,
      completionTokens: 120,
      cachedTokens: 1500,
      cost: 0.000412,
      calls: 1,
    });
  });

  it("returns null when the reply carries no usage", () => {
    expect(readUsage({ choices: [] })).toBeNull();
    expect(readUsage(null)).toBeNull();
    expect(readUsage("nonsense")).toBeNull();
  });

  it("survives a provider that reports tokens but not cost", () => {
    const usage = readUsage(response({ cost: undefined }));
    expect(usage).toMatchObject({ promptTokens: 3400, cost: 0 });
  });

  it("survives a provider that omits the cache breakdown", () => {
    const usage = readUsage(response({ prompt_tokens_details: undefined }));
    expect(usage).toMatchObject({ cachedTokens: 0 });
  });
});

describe("createUsageMeter", () => {
  it("adds up a run", () => {
    const meter = createUsageMeter();
    meter.record(readUsage(response()));
    meter.record(readUsage(response()));

    expect(meter.total()).toEqual({
      promptTokens: 6800,
      completionTokens: 240,
      cachedTokens: 3000,
      cost: 0.000824,
      calls: 2,
    });
  });

  it("ignores a call that reported nothing", () => {
    const meter = createUsageMeter();
    meter.record(null);

    expect(meter.total().calls).toBe(0);
  });

  it("says nothing at all when nothing was spent", () => {
    // A resumed run that had no work to do should not print a cost line.
    expect(createUsageMeter().summary()).toBe("");
  });

  it("reports the cache share, which is the one saving nobody can see", () => {
    // The 56-topic catalogue rides in the system prompt of every call. Whether
    // the provider is caching it is the difference between paying for it once
    // and paying for it 500 times.
    const meter = createUsageMeter();
    meter.record(readUsage(response()));

    const summary = meter.summary();
    expect(summary).toMatch(/1 call/);
    expect(summary).toMatch(/\$0\.0004/);
    expect(summary).toMatch(/44% cached/);
  });
});
