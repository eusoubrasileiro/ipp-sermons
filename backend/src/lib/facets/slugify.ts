/**
 * URL slugs for facet values, folded the same way the database folds text.
 *
 * Every facet in the UI is addressable (`/biblia/efesios/5`,
 * `/series/o-livro-dos-reis`), so the slug is a stable public identifier, not a
 * display detail. It has to survive the accents that Portuguese titles are full
 * of and the punctuation that sermon titles collect.
 *
 * NFD + combining-mark stripping is the same accent folding as the `unaccent`
 * dictionary behind the `pt_unaccent` search configuration, so a slug and a
 * lexical match agree on what "Efésios" reduces to.
 */
export function slugify(text: string): string {
  return (
    text
      .normalize("NFD")
      .replace(/[̀-ͯ]/g, "")
      .toLowerCase()
      // Keep digits: "1 Reis" and "2 Reis" are different books, and dropping the
      // numeral would silently merge them.
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
  );
}

/** Accent- and case-folded form for comparing two strings, without the dashes. */
export function fold(text: string): string {
  return text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/\s+/g, " ").trim();
}
