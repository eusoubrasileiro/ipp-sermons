import { describe, expect, it } from "vitest";
import { clientKey, createRateLimiter } from "../src/lib/rate-limit.ts";

/**
 * The only unauthenticated write this site has.
 *
 * `POST /api/suggestion` inserts straight from the request body, and the box it
 * runs on has about 2 GB free of 7.8. Nothing bounded how often a visitor could
 * do that, so the protection is here rather than in a dependency: one Map, a
 * fixed window, and a clock that is injected so the tests do not sleep.
 */

const HOUR = 60 * 60 * 1000;

/** A limiter whose clock is a variable, so a window can be crossed instantly. */
function atMinute(limit: number, windowMs = HOUR) {
  let now = 0;
  const limiter = createRateLimiter({ limit, windowMs, now: () => now });
  return { limiter, advance: (ms: number) => (now += ms) };
}

describe("createRateLimiter", () => {
  it("allows a visitor up to the limit", () => {
    const { limiter } = atMinute(3);
    expect([limiter.take("a"), limiter.take("a"), limiter.take("a")]).toEqual([true, true, true]);
  });

  it("refuses the one after that", () => {
    const { limiter } = atMinute(3);
    for (let i = 0; i < 3; i++) limiter.take("a");
    expect(limiter.take("a")).toBe(false);
  });

  it("counts each visitor separately", () => {
    // Otherwise the first person to hit the limit silences everybody else,
    // which turns a rate limit into an outage anyone can cause.
    const { limiter } = atMinute(1);
    expect(limiter.take("a")).toBe(true);
    expect(limiter.take("b")).toBe(true);
    expect(limiter.take("a")).toBe(false);
  });

  it("forgives once the window has passed", () => {
    const { limiter, advance } = atMinute(1);
    limiter.take("a");
    advance(HOUR + 1);
    expect(limiter.take("a")).toBe(true);
  });

  it("forgets visitors it has not heard from, so the map cannot grow forever", () => {
    // The map is the thing being protected as much as the table: a key per
    // address, kept for the life of the process, is its own slow leak.
    const { limiter, advance } = atMinute(1);
    for (let i = 0; i < 500; i++) limiter.take(`visitor-${i}`);
    expect(limiter.size()).toBe(500);

    advance(HOUR + 1);
    limiter.take("someone-new");
    expect(limiter.size()).toBe(1);
  });
});

describe("clientKey", () => {
  it("reads the first hop of X-Forwarded-For", () => {
    // Traefik terminates TLS, so every request reaches the app from inside the
    // Docker network. Without this every visitor shares one key and the limit
    // is a global one.
    expect(clientKey(new Headers({ "x-forwarded-for": "203.0.113.7, 10.0.0.2" }))).toBe(
      "203.0.113.7",
    );
  });

  it("falls back to the real-ip header Traefik also sets", () => {
    expect(clientKey(new Headers({ "x-real-ip": "203.0.113.9" }))).toBe("203.0.113.9");
  });

  it("gives every anonymous caller the same key when there is no header", () => {
    // Deliberately not "allow it": a request with no forwarded address is
    // either a direct hit on the container or a proxy that lost the header,
    // and sharing one bucket is the safe reading of both.
    expect(clientKey(new Headers())).toBe("unknown");
  });
});
