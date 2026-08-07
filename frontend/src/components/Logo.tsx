type LogoProps = { className?: string };

/**
 * The church's own pilgrim, vectorised from the site icon on ipperegrinos.com
 * (`potrace` over the 355px source, blue tile dropped).
 *
 * A silhouette rather than the outline drawing that was here before: the church
 * already has a mark, and an approximation of it is just a second, worse mark.
 * One filled path in `currentColor`, so it needs no second asset for dark mode
 * and follows `BrandIcons.tsx` in inlining brand marks rather than shipping
 * files. The viewBox is the figure's own bounds, so `h-*` alone sizes it.
 */
function PilgrimMark({ className = "h-9 w-auto" }: LogoProps) {
  return (
    <svg viewBox="0 0 30.88 46" className={className} aria-hidden="true" focusable="false">
      <title>Peregrinos</title>
      {/* evenodd, not nonzero: the gaps the trace produced — between the legs,
          under the arm, either side of the staff — are subpaths wound the same
          way as the body, and nonzero would fill them in. */}
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M0 45.9C-0 45.8 -0 45.5 0 45.1L0 44.4L4 44.4C7.4 44.3 7.9 44.3 8 44.2C8.1 44 9.2 40.1 9.2 39.9C9.2 39.8 9.3 39.7 9.3 39.6C9.5 39 10.1 37.4 10.4 36.7C11 34.9 11.6 33.7 12.1 32.7C13.1 30.9 13.5 28.1 13.2 24.9C13.1 24.1 13 23.2 13 22.8L13 22.1L12.4 22.2C11.6 22.3 9.8 22 9.1 21.6C8.4 21.2 7.3 19.7 7 18.7C6.7 18 6.7 15.1 6.9 14.2C7.2 12.8 7.2 12.7 7 11.8C6.7 10.3 6.8 9.1 7.3 8C7.7 7.3 7.7 7.2 7.3 7.2C6.9 7.2 6.9 7.1 8.3 5.6C9.7 4 12.3 4.4 14.1 6.3C14.6 6.9 14.8 7.1 14.9 7C15 7 15.3 6.9 15.6 6.9C16.7 6.6 16.8 5.9 15.9 2.9C15.4 1.1 17.4 -0.5 19.6 0C20.3 0.2 20.9 1 20.7 1.5C20.6 1.8 20.6 2 21.1 3.3C21.6 4.4 21.6 4.4 21.1 4.5C20.9 4.5 20.7 4.6 20.7 4.7C20.7 4.7 20.7 4.8 20.7 4.9C20.7 5 20.6 5.4 20.6 5.7C20.5 6.4 20.4 6.6 19.8 6.5L19.4 6.5L19.2 7.1C19.1 7.7 19.1 7.7 19.5 8.3C19.9 8.7 19.9 9 19.8 9.3C19.7 9.5 19.7 9.7 19.9 10.1C20.4 11.3 20.6 12.1 20.6 14.1C20.6 15.1 20.7 16.1 20.7 16.3C20.9 16.8 22.2 18.2 23.8 19.4C23.9 19.5 24.8 19.8 25 19.8C25.1 19.8 25.4 19 26 17.3C26.8 14.8 26.8 15 27.1 15C27.5 15.1 27.4 15.6 25.8 19.5C25.5 20.1 25.5 20.1 26 20.5L26.3 20.8L26.2 21.1C26 21.4 25.8 21.5 25.4 21.8C24.8 22.1 24.9 21.9 24.1 24C23.9 24.5 23.7 25.1 23.6 25.2C23.6 25.4 23.4 25.7 23.4 26C23.3 26.3 22.9 27.3 22.5 28.2L21.9 29.9L22.1 30.4C22.3 30.8 22.4 31.3 22.7 32.7C22.8 33.1 23.3 36.1 23.4 37.5C23.5 38 23.6 38.6 23.6 38.8C23.7 39.4 23.8 40.8 23.8 41.2C23.7 42.1 25.6 43.1 27.5 43.2C28.5 43.3 28.5 43.3 28.6 44L28.7 44.3L29.8 44.4L30.9 44.4L30.9 45.2L30.9 46L15.5 46C2.7 46 0.1 46 0 45.9ZM16.2 43.6C16.4 43.2 16.6 42.6 16.7 42.3C16.8 42 17 41.5 17.1 41.1C17.2 40.8 17.5 40 17.8 39.4C18 38.7 18.5 37.5 18.8 36.6C19.3 35.1 19.4 34.8 19.3 34.3C19.1 33.8 19 33.1 18.9 32.4C18.7 31.5 18.2 30.5 17.6 29.9C17 29.3 16.9 29.3 16.7 30.1C16.5 30.8 15.8 32.5 14.9 34.5C14.7 35.2 14.3 36 14.1 36.4C14 36.8 13.6 37.7 13.3 38.3C12.2 40.5 12 41.7 12.3 42.5C13.1 44 13.6 44.4 15 44.3L15.9 44.3L16.2 43.6ZM21.3 44.3C21.4 44.2 21.1 42.5 20.4 39.4C20.2 38.8 20 37.9 19.9 37.5C19.8 37.1 19.7 36.7 19.7 36.5C19.7 36.3 19.5 36.3 19.5 36.5C19.4 36.7 17.9 40.7 17.8 40.8C17.7 41.2 16.6 44.1 16.6 44.2C16.6 44.4 21 44.5 21.3 44.3ZM21.8 28.4C21.9 28.2 22.2 27.2 22.6 26.1C23 25.1 23.4 24 23.5 23.8C24.1 22.3 24.2 21.9 24.1 21.8C23.8 21.3 20.8 18.8 20.1 18.5C19.5 18.2 19.5 18.2 19.3 19.9C19.1 22.2 19.5 23.5 21.1 27.8C21.6 29 21.6 29 21.8 28.4Z"
      />
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
        <PilgrimMark className="h-12 w-auto shrink-0 text-primary sm:h-14" />
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
