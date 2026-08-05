import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { type BrowseSermon, browseSermons } from "../api.ts";
import { SermonListItem } from "../components/SermonListItem.tsx";

/**
 * A filtered listing: what every index page links to.
 *
 * No query, no ranking -- the person already chose the facet, so the job is
 * simply to show what is in it, newest first (or in course order for a series).
 */
export function BrowseResults({
  titulo,
  subtitulo,
  query,
  voltar,
}: {
  titulo: string;
  subtitulo?: string | undefined;
  /** Already serialised: a primitive, so the fetch effect depends on it
   * directly instead of on an object literal rebuilt every render. */
  query: string;
  voltar: { to: string; label: string };
}) {
  const [sermons, setSermons] = useState<BrowseSermon[]>([]);
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState<"loading" | "done" | "error">("loading");

  useEffect(() => {
    let alive = true;
    setStatus("loading");

    browseSermons(query)
      .then((res) => {
        if (!alive) return;
        setSermons(res.sermons);
        setTotal(res.total);
        setStatus("done");
      })
      .catch(() => alive && setStatus("error"));

    return () => {
      alive = false;
    };
  }, [query]);

  return (
    <div>
      <Link to={voltar.to} className="text-sm text-muted-foreground hover:text-foreground">
        ← {voltar.label}
      </Link>
      <h1 className="mt-1 text-xl font-bold tracking-tight sm:text-2xl">{titulo}</h1>
      {subtitulo ? <p className="text-sm text-muted-foreground">{subtitulo}</p> : null}

      <p aria-live="polite" className="mt-1 text-xs text-muted-foreground">
        {status === "loading" ? "Carregando…" : null}
        {status === "done" ? `${total} ${total === 1 ? "sermão" : "sermões"}` : null}
        {status === "error" ? "Não foi possível carregar." : null}
      </p>

      <div className="mt-3">
        {sermons.map((s) => (
          <SermonListItem key={s.id} sermon={s} />
        ))}
      </div>

      {status === "done" && total === 0 ? (
        <p className="mt-4 text-sm text-muted-foreground">Nenhum sermão aqui ainda.</p>
      ) : null}
    </div>
  );
}
