import { NavLink } from "react-router-dom";

/**
 * The persistent tab bar under the search box.
 *
 * Search stays the hero -- 456 sermons is small enough that most people will
 * type -- but the archive was invisible to anyone who did not already know
 * what to ask for. These are the second door.
 *
 * Scrolls horizontally on a phone rather than wrapping to two lines, so the
 * bar never pushes the results below the fold.
 */
const TABS = [
  { to: "/", label: "Buscar", end: true },
  { to: "/temas", label: "Temas" },
  { to: "/biblia", label: "Bíblia" },
  { to: "/series", label: "Séries" },
  { to: "/pregadores", label: "Pregadores" },
  { to: "/datas", label: "Datas" },
];

export function FacetNav() {
  return (
    <nav
      aria-label="Navegar no acervo"
      className="-mx-4 mt-3 overflow-x-auto border-b border-border px-4"
    >
      <ul className="flex min-w-max gap-1 text-sm">
        {TABS.map((tab) => (
          <li key={tab.to}>
            <NavLink
              to={tab.to}
              end={tab.end}
              className={({ isActive }) =>
                [
                  "inline-flex min-h-11 items-center border-b-2 px-3 transition",
                  isActive
                    ? "border-primary font-medium text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground",
                ].join(" ")
              }
            >
              {tab.label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}
