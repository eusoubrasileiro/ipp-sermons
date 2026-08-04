export default {
  "*.{ts,tsx,js}": () => ["pnpm lint", "pnpm -r typecheck", "pnpm -r test"],
};
