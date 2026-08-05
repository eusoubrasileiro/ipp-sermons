import { useParams } from "react-router-dom";
import { type FacetTree, toBrowseQuery } from "../api.ts";
import type { FacetGroup } from "../components/FacetIndex.tsx";
import { FacetIndexPage } from "../components/FacetIndexPage.tsx";
import { BrowseResults } from "./BrowseResults.tsx";

const MONTHS = [
  "Janeiro",
  "Fevereiro",
  "Março",
  "Abril",
  "Maio",
  "Junho",
  "Julho",
  "Agosto",
  "Setembro",
  "Outubro",
  "Novembro",
  "Dezembro",
];

/**
 * Year, then month. No decades -- the corpus starts in 2019, and "the sermon
 * from last Sunday" is the commonest request this page serves, so the newest
 * year comes first.
 */
function toGroups(facets: FacetTree): FacetGroup[] {
  return facets.datas.map((year) => ({
    label: String(year.ano),
    entries: year.meses
      .sort((a, b) => b.mes - a.mes)
      .map((m) => ({
        to: `/datas/${year.ano}/${m.mes}`,
        label: MONTHS[m.mes - 1] ?? String(m.mes),
        total: m.total,
      })),
  }));
}

const lastDay = (ano: number, mes: number) => new Date(Date.UTC(ano, mes, 0)).getUTCDate();

export function DatesPage() {
  const { ano, mes } = useParams();

  if (ano) {
    const year = Number.parseInt(ano, 10);
    const month = mes ? Number.parseInt(mes, 10) : null;
    const pad = (n: number) => String(n).padStart(2, "0");

    return (
      <BrowseResults
        titulo={month ? `${MONTHS[month - 1]} de ${year}` : String(year)}
        query={toBrowseQuery({
          de: month ? `${year}-${pad(month)}-01` : `${year}-01-01`,
          ate: month ? `${year}-${pad(month)}-${pad(lastDay(year, month))}` : `${year}-12-31`,
        })}
        voltar={{ to: "/datas", label: "Datas" }}
      />
    );
  }

  return <FacetIndexPage agrupar={toGroups} />;
}
