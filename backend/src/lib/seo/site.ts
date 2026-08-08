/**
 * Where this site lives, as far as a crawler is concerned.
 *
 * A canonical URL and an Open Graph URL have to be absolute, and the request's
 * own `Host` is not a safe source for them: behind Traefik the container sees
 * an internal origin, and a crawler that reaches the site by any other name
 * would be told that name is canonical. So it is configuration, with the
 * production origin as the default so no deployment change is needed to get it
 * right.
 */
export const SITE_URL = (process.env.PUBLIC_BASE_URL ?? "https://ipp-sermons.amiticia.cc").replace(
  /\/+$/,
  "",
);

/**
 * Where the built SPA is. Relative to the working directory, matching the
 * `serveStatic({ root: "./public" })` the shell is otherwise served by.
 */
export const PUBLIC_DIR = process.env.PUBLIC_DIR ?? "./public";
