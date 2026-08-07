type LogoProps = { className?: string };

/**
 * The church's own pilgrim, vectorised from the site icon on ipperegrinos.com
 * (`potrace` over the 355px source, blue tile dropped).
 *
 * The source is antialiased, and that turns out to be the whole trick:
 * thresholding to black-and-white first and *then* upscaling makes potrace fit
 * curves to a pixel staircase — 19k characters of path, with the backpack
 * melted into the body. Upscaling the grey and thresholding after keeps the
 * partial pixels that put each edge on a sub-pixel boundary, which is both
 * faithful and four times smaller.
 *
 * A silhouette rather than the outline drawing that was here before: the church
 * already has a mark, and an approximation of it is just a second, worse mark.
 * One filled path in `currentColor`, so it needs no second asset for dark mode
 * and follows `BrandIcons.tsx` in inlining brand marks rather than shipping
 * files. The viewBox is the figure's own bounds, so `h-*` alone sizes it.
 */
function PilgrimMark({ className = "h-9 w-auto" }: LogoProps) {
  return (
    <svg viewBox="0 0 31.02 46" className={className} aria-hidden="true" focusable="false">
      <title>Peregrinos</title>
      {/* evenodd, not nonzero: the gaps the trace produced — between the legs,
          under the arm, either side of the staff — are subpaths wound the same
          way as the body, and nonzero would fill them in. */}
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M-0 45.22L-0 44.44L0.16 44.42C0.25 44.4 2.03 44.39 4.12 44.39C7.8 44.39 7.92 44.39 7.99 44.31C8.06 44.23 8.26 43.6 8.34 43.21C8.36 43.13 8.4 43 8.43 42.93C8.46 42.86 8.57 42.49 8.67 42.12C8.77 41.75 8.92 41.2 9.02 40.9C9.11 40.6 9.2 40.25 9.21 40.12C9.25 39.83 9.51 39.02 9.76 38.44C9.86 38.21 10.01 37.84 10.08 37.62C11.25 34.4 11.4 34.05 12.19 32.65C12.75 31.65 12.93 31.08 13.11 29.64C13.31 28.03 13.35 26.19 13.21 24.96C13.16 24.52 13.09 23.79 13.07 23.35C13 22.25 12.96 22.08 12.77 22.19C12.34 22.46 10.13 22.15 9.21 21.7C8.35 21.27 7.21 19.74 6.92 18.62C6.67 17.62 6.72 15.05 7.01 13.92C7.27 12.96 7.26 12.68 6.98 11.48C6.76 10.52 6.88 9.14 7.27 8.25C7.65 7.39 7.65 7.38 7.37 7.34C6.97 7.27 6.96 7.2 7.31 6.85C7.45 6.71 7.71 6.4 7.89 6.16C8.23 5.73 8.75 5.22 9.02 5.06C9.53 4.76 10.16 4.59 10.54 4.66C10.67 4.68 10.92 4.72 11.09 4.74C12.42 4.94 13.2 5.41 14.39 6.76C14.74 7.16 14.71 7.15 15.38 6.99C16.33 6.75 16.76 6.28 16.58 5.65C16.46 5.22 16.22 4.24 16.1 3.68C16.03 3.39 15.95 3.06 15.91 2.94C15.79 2.58 15.81 1.83 15.94 1.52C16.24 0.84 17.13 0.23 17.96 0.13C18.17 0.11 18.38 0.07 18.42 0.05C18.62 -0.06 19.59 -0.01 19.86 0.12C20.29 0.32 20.34 0.37 20.58 0.88L20.82 1.35L20.75 1.56C20.63 1.96 20.65 2.05 21.23 3.59C21.56 4.47 21.55 4.51 21.13 4.56C20.74 4.61 20.71 4.67 20.65 5.36C20.55 6.54 20.39 6.77 19.8 6.6C19.47 6.51 19.42 6.54 19.31 6.93C19.07 7.75 19.07 7.78 19.54 8.33C19.89 8.73 19.95 8.98 19.8 9.33C19.7 9.56 19.71 9.59 19.97 10.24C20.52 11.6 20.76 13.32 20.64 15.03C20.55 16.42 20.69 16.72 22.09 18.04C23.55 19.42 23.89 19.64 24.87 19.85C25.02 19.88 25.11 19.78 25.24 19.4C25.36 19.06 25.69 18.15 25.9 17.57C25.98 17.35 26.05 17.15 26.05 17.13C26.05 17.1 26.14 16.85 26.25 16.57C26.36 16.28 26.51 15.88 26.58 15.66C26.78 15.04 26.89 14.96 27.22 15.16C27.48 15.32 27.47 15.39 27.03 16.45C26.75 17.16 26.16 18.64 25.83 19.49C25.53 20.27 25.52 20.21 26.07 20.66C26.48 21 26.21 21.44 25.34 21.89C24.77 22.17 24.84 22.07 24.46 23.15C24.33 23.51 24.1 24.15 23.93 24.57C23.77 24.99 23.54 25.62 23.41 25.98C23.29 26.34 23.09 26.87 22.98 27.16C22.86 27.45 22.71 27.84 22.64 28.03C21.89 30.03 21.91 29.92 22.16 30.54C22.39 31.16 22.47 31.41 22.61 32.14C22.68 32.51 22.75 32.82 22.76 32.84C22.81 32.92 23.21 35.66 23.33 36.74C23.37 37.08 23.41 37.48 23.43 37.64C23.46 37.79 23.5 38.11 23.52 38.33C23.55 38.56 23.6 38.85 23.63 38.96C23.72 39.3 23.8 40.14 23.81 40.98C23.82 41.98 23.81 41.96 24.69 42.4C25.95 43.03 26.88 43.3 27.78 43.3C28.44 43.3 28.58 43.43 28.68 44.11C28.73 44.4 28.69 44.39 29.79 44.39L30.76 44.39L30.89 44.53C31.03 44.69 31.08 45.53 30.96 45.86L30.91 46L15.46 46L0 46L-0 45.22ZM15.96 44.29C15.99 44.24 16.09 43.99 16.18 43.73C16.27 43.48 16.38 43.18 16.43 43.06C16.48 42.94 16.61 42.58 16.73 42.25C16.99 41.5 17.25 40.81 17.62 39.86C17.78 39.44 17.97 38.93 18.03 38.73C18.1 38.52 18.29 38.01 18.45 37.59C18.61 37.17 18.8 36.64 18.88 36.42C18.96 36.19 19.1 35.82 19.18 35.6C19.29 35.33 19.34 35.13 19.34 34.98C19.34 34.79 19.11 33.49 19.06 33.41C19.05 33.4 19.02 33.21 18.99 32.99C18.82 31.65 18.38 30.66 17.66 29.97C17.07 29.4 16.9 29.4 16.75 29.99C16.67 30.35 16.3 31.29 15.63 32.89C15.52 33.17 15.42 33.42 15.42 33.43C15.42 33.45 15.32 33.69 15.2 33.96C15.08 34.23 14.93 34.58 14.88 34.73C14.82 34.88 14.64 35.3 14.48 35.68C14.32 36.05 14.12 36.54 14.03 36.77C13.79 37.36 13.77 37.4 12.96 39.09C12.74 39.55 12.29 40.9 12.17 41.43C12.06 41.95 12.14 42.24 12.68 43.19C12.89 43.57 13.43 44.09 13.77 44.26L14.05 44.39L14.97 44.39C15.88 44.39 15.9 44.39 15.96 44.29ZM21.34 44.34C21.42 44.25 21.28 43.32 21 42.1C20.95 41.91 20.86 41.52 20.8 41.23C20.68 40.68 20.44 39.65 20.19 38.62C19.97 37.71 19.83 37.08 19.77 36.8C19.67 36.25 19.55 36.31 19.3 37.03C19.21 37.3 19.03 37.79 18.91 38.12C18.78 38.44 18.59 38.94 18.48 39.23C18.37 39.51 18.22 39.95 18.14 40.18C18.06 40.42 17.94 40.73 17.88 40.86C17.79 41.07 17.25 42.49 16.83 43.65C16.66 44.1 16.64 44.29 16.73 44.35C16.83 44.41 21.27 44.4 21.34 44.34ZM21.78 28.54C21.83 28.41 21.94 28.11 22.02 27.88C22.1 27.65 22.3 27.12 22.46 26.7C22.78 25.87 22.95 25.39 23.32 24.34C23.45 23.97 23.58 23.62 23.6 23.58C23.63 23.53 23.77 23.17 23.91 22.76C24.22 21.92 24.21 21.9 23.9 21.64C23.8 21.56 23.4 21.2 23.02 20.84C20.86 18.82 19.8 18.15 19.5 18.61C19.33 18.87 19.23 21.64 19.36 22.46C19.45 23.08 19.81 24.27 20.39 25.9C21.49 28.95 21.57 29.1 21.78 28.54Z"
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
