/**
 * Colors are HSL CSS variables (src/index.css) surfaced as semantic Tailwind
 * names, so components never reference a raw palette shade and light/dark stay
 * in one place.
 */
const token = (name) => `hsl(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: token("background"),
        foreground: token("foreground"),
        border: token("border"),
        input: token("input"),
        ring: token("ring"),
        card: { DEFAULT: token("card"), foreground: token("card-foreground") },
        primary: { DEFAULT: token("primary"), foreground: token("primary-foreground") },
        secondary: { DEFAULT: token("secondary"), foreground: token("secondary-foreground") },
        /* `gold` is contrast-checked for text; `gold-rule` is the church's own
           brighter gold, for dividers and icons only. */
        gold: { DEFAULT: token("gold"), rule: token("gold-rule") },
        muted: { DEFAULT: token("muted"), foreground: token("muted-foreground") },
        accent: { DEFAULT: token("accent"), foreground: token("accent-foreground") },
        destructive: { DEFAULT: token("destructive"), foreground: token("destructive-foreground") },
        warning: token("warning"),
        highlight: { DEFAULT: token("highlight"), foreground: token("highlight-foreground") },
        soundcloud: { DEFAULT: token("soundcloud"), foreground: token("soundcloud-foreground") },
        spotify: { DEFAULT: token("spotify"), foreground: token("spotify-foreground") },
      },
      fontFamily: {
        display: ['"Playfair Display"', "Georgia", '"Times New Roman"', "serif"],
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
};
