import { Link } from "react-router-dom";

/**
 * The template behind all five index pages.
 *
 * Desiring God renders /topics, /scripture, /series, /authors and /dates with
 * one layout — a group label on the left, its entries with counts on the
 * right — and that is most of the value of those pages for very little code.
 * Ours differs in one way: at 456 sermons a facet with nothing in it is noise,
 * so empty entries never render at all.
 *
 * Two columns on a desktop; on a phone the group label becomes a sticky header
 * above its own entries, because a 120px label column leaves nothing for the
 * names.
 */
export type FacetEntry = {
  to: string;
  label: string;
  /** Second line under the label — a series' event, a preacher's honorific. */
  detail?: string | undefined;
  total: number;
  /** Rendered indented, as chapters are under their book. */
  children?: FacetEntry[] | undefined;
};

export type FacetGroup = { label: string; entries: FacetEntry[] };

function Row({ entry, nested }: { entry: FacetEntry; nested?: boolean }) {
  return (
    <Link
      to={entry.to}
      // The count is not part of the name: read as one phrase it becomes
      // "Genesis seventy-three" rather than "Genesis, 73 sermons".
      aria-label={`${entry.label}, ${entry.total} ${entry.total === 1 ? "sermão" : "sermões"}`}
      className={[
        "flex min-h-11 items-center justify-between gap-3 border-b border-border px-1 py-2",
        "transition hover:bg-accent hover:text-accent-foreground",
        nested ? "pl-5 text-sm" : "font-medium",
      ].join(" ")}
    >
      <span className="min-w-0">
        <span className="block truncate">{entry.label}</span>
        {entry.detail ? (
          <span className="block truncate text-xs font-normal text-muted-foreground">
            {entry.detail}
          </span>
        ) : null}
      </span>
      <span aria-hidden="true" className="shrink-0 tabular-nums text-sm text-muted-foreground">
        ({entry.total})
      </span>
    </Link>
  );
}

export function FacetIndex({ groups }: { groups: FacetGroup[] }) {
  const populated = groups.filter((g) => g.entries.length > 0);

  if (populated.length === 0) {
    return (
      <p className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground">
        Nada por aqui ainda.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      {populated.map((group) => (
        <section key={group.label} className="sm:flex sm:gap-6">
          <h2 className="sticky top-0 z-[1] -mx-4 bg-background px-4 py-2 text-sm font-semibold uppercase tracking-wide text-muted-foreground sm:static sm:mx-0 sm:w-40 sm:shrink-0 sm:bg-transparent sm:px-0 sm:normal-case sm:tracking-normal sm:text-base sm:text-foreground">
            {group.label}
          </h2>
          <div className="min-w-0 flex-1">
            {group.entries.map((entry) => (
              <div key={entry.to}>
                <Row entry={entry} />
                {entry.children?.map((child) => (
                  <Row key={child.to} entry={child} nested />
                ))}
              </div>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
