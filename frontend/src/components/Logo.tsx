type LogoProps = { className?: string };

/**
 * The church's pilgrim mark — a traveller with a staff and pack, over the rule
 * that underlines the wordmark on ipperegrinos.com.
 *
 * Redrawn as a path rather than shipping their PNG: it inherits `currentColor`,
 * so it works on both colour schemes without a second asset, and it follows the
 * precedent set in `BrandIcons.tsx` that brand marks are inlined, not files.
 */
function PilgrimMark({ className = "h-9 w-9" }: LogoProps) {
  return (
    <svg viewBox="0 0 48 48" className={className} aria-hidden="true" focusable="false">
      <title>Peregrinos</title>
      {/* Drawn as strokes, not one filled outline: at 40px a silhouette turns to
          mud, while round-capped strokes keep the stride and the staff legible. */}
      <g
        fill="none"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="21" cy="8.5" r="3.4" fill="currentColor" stroke="none" />
        {/* pack on the back */}
        <path
          d="M13.5 14.5h4.5v7.5h-4.5z"
          fill="currentColor"
          stroke="currentColor"
          strokeWidth="2"
        />
        {/* torso, then the two legs mid-stride */}
        <path d="M20.5 13.5 19 24" />
        <path d="M19 24l5.5 7.5.5 9" />
        <path d="M19 24l-4.5 8-2 8.5" />
        {/* arm reaching the staff */}
        <path d="M20 17.5 27 21" />
        {/* staff, angled as a walking stick rather than a post */}
        <path d="M30 5.5 26.5 42" strokeWidth="2.2" />
      </g>
    </svg>
  );
}

/**
 * Full lockup for the page header: the mark, the church name in the display
 * serif, and the rule in the brand's own gold.
 */
export function Wordmark({ className = "" }: LogoProps) {
  return (
    // The rule runs under the mark as well as the words, as it does on the
    // church's own lockup, which is what makes the two read as one thing.
    <span className={`inline-block ${className}`}>
      <span className="flex items-end gap-3">
        <PilgrimMark className="h-11 w-11 shrink-0 text-primary sm:h-14 sm:w-14" />
        <span className="min-w-0 pb-0.5">
          <span className="block font-display text-[0.7rem] uppercase tracking-[0.18em] text-muted-foreground sm:text-xs">
            Igreja Presbiteriana
          </span>
          <span className="block font-display text-2xl font-bold leading-none tracking-wide text-primary sm:text-3xl">
            Peregrinos
          </span>
        </span>
      </span>
      <span className="mt-1.5 block h-0.5 w-full rounded-full bg-gold-rule" />
    </span>
  );
}
