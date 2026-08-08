import { describe, expect, it } from "vitest";
import { formatDuration } from "./format.ts";

/**
 * `stripLeadingDate` and `formatDate` moved to `@ipp/shared` when the server
 * started rendering the same heading; their tests moved with them.
 */

describe("formatDuration", () => {
  it("drops a leading zero hour", () => {
    expect(formatDuration("0:45:49")).toBe("45:49");
    expect(formatDuration("1:05:50")).toBe("1:05:50");
  });
});
