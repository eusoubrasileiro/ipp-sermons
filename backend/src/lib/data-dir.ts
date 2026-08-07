import { join } from "node:path";

/**
 * Where the corpus lives: `data/` at the repo root, or wherever `CORPUS_DIR`
 * points.
 *
 * `CORPUS_DIR` is what the production image sets (`/app/data`), because the
 * relative default resolves against the module rather than the working
 * directory and the container's layout is not the repo's. Six scripts declared
 * this identically; one copy means one place for that to be wrong.
 *
 * `../../../data` is the same distance from `src/lib` as from `src/scripts`, so
 * moving it here did not change what it resolves to.
 */
export const DATA_DIR = process.env.CORPUS_DIR ?? join(import.meta.dirname, "../../../data");
