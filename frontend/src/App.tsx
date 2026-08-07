import { Route, Routes } from "react-router-dom";
import { FacetNav } from "./components/FacetNav.tsx";
import { Wordmark } from "./components/Logo.tsx";
import { BiblePage } from "./pages/BiblePage.tsx";
import { DatesPage } from "./pages/DatesPage.tsx";
import { PreachersPage } from "./pages/PreachersPage.tsx";
import { SearchPage } from "./pages/SearchPage.tsx";
import { SeriesPage } from "./pages/SeriesPage.tsx";
import { TopicsPage } from "./pages/TopicsPage.tsx";

/**
 * The shell: header, tab bar, and whichever page the URL names.
 *
 * Every facet is addressable -- /biblia/efesios/5, /series/o-livro-dos-reis --
 * so a member can send a passage or a whole course to someone in WhatsApp and
 * have it open where they left it. That is the reason for a router in a site
 * this small.
 */
export function App() {
  return (
    <div className="min-h-dvh">
      <div className="mx-auto max-w-3xl px-4 pb-12">
        <header className="pt-6 sm:pt-10">
          <Wordmark />
          <p className="mt-3 text-sm text-muted-foreground sm:text-base">
            Busque nos sermões por tema, passagem bíblica ou pregador.
          </p>
        </header>

        <FacetNav />

        <main className="mt-5">
          <Routes>
            <Route path="/" element={<SearchPage />} />
            <Route path="/temas" element={<TopicsPage />} />
            <Route path="/temas/:slug" element={<TopicsPage />} />
            <Route path="/biblia" element={<BiblePage />} />
            <Route path="/biblia/:livro" element={<BiblePage />} />
            <Route path="/biblia/:livro/:capitulo" element={<BiblePage />} />
            <Route path="/series" element={<SeriesPage />} />
            <Route path="/series/:slug" element={<SeriesPage />} />
            <Route path="/pregadores" element={<PreachersPage />} />
            <Route path="/pregadores/:slug" element={<PreachersPage />} />
            <Route path="/datas" element={<DatesPage />} />
            <Route path="/datas/:ano" element={<DatesPage />} />
            <Route path="/datas/:ano/:mes" element={<DatesPage />} />
            <Route path="*" element={<SearchPage />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
