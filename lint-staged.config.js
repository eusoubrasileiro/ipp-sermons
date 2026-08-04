export default {
  "*.{ts,tsx,js}": () => [
    "pnpm lint",
    "pnpm --filter @project/backend exec tsc --noEmit",
    "pnpm --filter @project/frontend exec tsc --noEmit",
    "pnpm test",
    "pnpm test:frontend",
  ],
};
