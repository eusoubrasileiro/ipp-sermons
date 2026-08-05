import { fold } from "./slugify.ts";

/**
 * Groups the raw series names the title parser produced into candidate courses.
 *
 * This is the cheap, deterministic first pass of series canonicalisation. It
 * catches the spelling drift a six-year corpus accumulates -- "Atribututos de
 * Deus" beside "Atributos de Deus" -- without spending a token. What it cannot
 * decide (which spelling is correct, what kind of course this is, whether two
 * differently-named courses are really one) is left to the LLM adjudication
 * pass, which receives these clusters as its input.
 *
 * Mirrors the rapidfuzz `token_set_ratio` approach already used for preacher
 * names in `tools/corpus-update/postprocess.py`.
 */
export type NameEntry = { name: string; count: number };

export type NameCluster = {
  /** The most-used spelling. Provisional: the LLM stage may replace it. */
  provisional: string;
  members: string[];
  count: number;
};

const tokens = (text: string): string[] =>
  fold(text)
    .replace(/[^a-z0-9]+/g, " ")
    .split(" ")
    .filter((t) => t.length > 0);

function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      row[j] = Math.min((row[j - 1] ?? 0) + 1, (prev[j] ?? 0) + 1, (prev[j - 1] ?? 0) + cost);
    }
    prev = row;
  }
  return prev[b.length] ?? 0;
}

const ratio = (a: string, b: string): number => {
  const longest = Math.max(a.length, b.length);
  if (longest === 0) return 1;
  return 1 - levenshtein(a, b) / longest;
};

/**
 * How alike two course names are, from 0 to 1.
 *
 * Compares both the folded strings and their sorted token sets, taking the
 * better of the two, so that a reordering ("Bispos e Pastores" vs "Pastores e
 * Bispos") scores as highly as a typo.
 */
export function similarity(a: string, b: string): number {
  const foldedA = tokens(a).join(" ");
  const foldedB = tokens(b).join(" ");
  const sortedA = [...new Set(tokens(a))].sort().join(" ");
  const sortedB = [...new Set(tokens(b))].sort().join(" ");
  return Math.max(ratio(foldedA, foldedB), ratio(sortedA, sortedB));
}

const ROMAN_VALUES: Record<string, number> = {
  i: 1,
  ii: 2,
  iii: 3,
  iv: 4,
  v: 5,
  vi: 6,
  vii: 7,
  viii: 8,
  ix: 9,
  x: 10,
};

/**
 * The ordinals a name carries, in order, arabic and roman alike.
 *
 * Load-bearing guard, and it has to cover both numeral systems:
 *
 * - "CFW 23" and "CFW 28" differ by one character in six and score 0.83 on any
 *   string metric, close enough that a slightly looser threshold collapses
 *   twenty-eight chapters of the Confession into one series.
 * - "IV Conferência Peregrinos" and "V Conferência Peregrinos" score higher
 *   still, and are five different annual conferences. An arabic-only signature
 *   merged all five into a single seventeen-sermon blob.
 *
 * Two names whose ordinals differ are never the same course, whatever they
 * look like to a string metric.
 */
export function digitSignature(name: string): string {
  const parts: string[] = [];
  for (const token of tokens(name)) {
    if (/^\d+$/.test(token)) parts.push(String(Number.parseInt(token, 10)));
    else if (token in ROMAN_VALUES) parts.push(String(ROMAN_VALUES[token]));
  }
  return parts.join(",");
}

const DEFAULT_THRESHOLD = 0.88;

export function clusterNames(entries: NameEntry[], threshold = DEFAULT_THRESHOLD): NameCluster[] {
  // Union-find, so that a~b and b~c land together even when a and c alone fall
  // short of the threshold.
  const parent = entries.map((_, i) => i);
  const find = (i: number): number => {
    let root = i;
    while (parent[root] !== root) root = parent[root] as number;
    return root;
  };
  const union = (i: number, j: number): void => {
    const [a, b] = [find(i), find(j)];
    if (a !== b) parent[a] = b;
  };

  for (let i = 0; i < entries.length; i++) {
    for (let j = i + 1; j < entries.length; j++) {
      const left = entries[i] as NameEntry;
      const right = entries[j] as NameEntry;
      if (digitSignature(left.name) !== digitSignature(right.name)) continue;
      if (similarity(left.name, right.name) >= threshold) union(i, j);
    }
  }

  const groups = new Map<number, NameEntry[]>();
  for (let i = 0; i < entries.length; i++) {
    const root = find(i);
    const group = groups.get(root) ?? [];
    group.push(entries[i] as NameEntry);
    groups.set(root, group);
  }

  return [...groups.values()]
    .map((group) => {
      const ranked = [...group].sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
      return {
        provisional: (ranked[0] as NameEntry).name,
        members: ranked.map((e) => e.name),
        count: group.reduce((sum, e) => sum + e.count, 0),
      };
    })
    .sort((a, b) => b.count - a.count || a.provisional.localeCompare(b.provisional));
}
