/**
 * A fixed-window counter, in memory, for the one write a visitor can make.
 *
 * `POST /api/suggestion` inserts straight from the request body with no account
 * behind it, and the VPS has ~2 GB free of 7.8. That is the whole threat: not
 * an attack, just nothing stopping a loop.
 *
 * In memory rather than in Postgres because the write being throttled is the
 * one we do not want to make, so spending a query to decide is backwards. One
 * container serves the whole site, so there is nothing to share state with; if
 * that ever stops being true this becomes wrong, and the deployment section is
 * where that would be decided.
 *
 * A fixed window rather than a token bucket: the limit here is "a handful an
 * hour", where the burst behaviour the bucket buys is not worth the arithmetic.
 */

export type RateLimiter = {
  /** True when this caller may proceed, and counts the attempt. */
  take(key: string): boolean;
  /** How many callers are being remembered. Exposed for the leak test. */
  size(): number;
};

type RateLimitOptions = {
  limit: number;
  windowMs: number;
  /** Injected so tests can cross a window without sleeping. */
  now?: () => number;
};

export function createRateLimiter({
  limit,
  windowMs,
  now = Date.now,
}: RateLimitOptions): RateLimiter {
  const seen = new Map<string, { count: number; resets: number }>();

  return {
    take(key) {
      const at = now();

      // Sweep on write rather than on a timer: without it the map keeps a
      // key per address for the life of the process, which is a slow leak on
      // the box this exists to protect.
      for (const [other, window] of seen) {
        if (window.resets <= at) seen.delete(other);
      }

      const window = seen.get(key);
      if (!window || window.resets <= at) {
        seen.set(key, { count: 1, resets: at + windowMs });
        return true;
      }
      if (window.count >= limit) return false;
      window.count += 1;
      return true;
    },
    size: () => seen.size,
  };
}

/**
 * Who is asking, as far as the container can tell.
 *
 * Traefik terminates TLS and proxies from inside the Docker network, so the
 * socket address is the router's every time. The first hop of
 * `X-Forwarded-For` is the visitor; the rest are proxies.
 *
 * A request with neither header shares one bucket with every other such
 * request. That is the safe reading: it is either a direct hit on the container
 * or a proxy that dropped the header, and neither deserves its own allowance.
 */
export function clientKey(headers: Headers): string {
  const forwarded = headers.get("x-forwarded-for");
  if (forwarded) {
    const first = forwarded.split(",")[0]?.trim();
    if (first) return first;
  }
  return headers.get("x-real-ip")?.trim() || "unknown";
}
