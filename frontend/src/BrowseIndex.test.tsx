import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { FACETS, routeTo } from "./browse-fixtures.tsx";
import { okResponse } from "./test-fixtures.tsx";

/**
 * The five index pages: grouping, ordering and counts.
 */
describe("navegação por facetas", () => {
  it("mostra as abas do acervo", () => {
    routeTo("/");
    const nav = screen.getByRole("navigation", { name: /navegar no acervo/i });
    for (const label of ["Buscar", "Temas", "Bíblia", "Séries", "Pregadores", "Datas"]) {
      expect(within(nav).getByRole("link", { name: label })).toBeInTheDocument();
    }
  });

  it("marca a aba atual para leitores de tela", async () => {
    routeTo("/biblia");
    const current = await screen.findByRole("link", { name: "Bíblia", current: "page" });
    expect(current).toBeInTheDocument();
  });
});

describe("/biblia", () => {
  it("agrupa por testamento em ordem canônica", async () => {
    routeTo("/biblia");
    await screen.findByText("Antigo Testamento");
    expect(screen.getByText("Novo Testamento")).toBeInTheDocument();

    // Canonical order, never alphabetical: Genesis precedes Efesios.
    const links = screen.getAllByRole("link").map((l) => l.textContent ?? "");
    const genesis = links.findIndex((t) => t.startsWith("Gênesis"));
    const efesios = links.findIndex((t) => t.startsWith("Efésios"));
    expect(genesis).toBeLessThan(efesios);
  });

  it("lista os capítulos que têm sermões e conta cada um", async () => {
    routeTo("/biblia");
    expect(await screen.findByRole("link", { name: /Capítulo 1, 6 sermões/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Capítulo 3, 4 sermões/i })).toBeInTheDocument();
    // Chapter 2 has none and must not appear.
    expect(screen.queryByRole("link", { name: /Capítulo 2,/i })).not.toBeInTheDocument();
  });

  it("abre direto na passagem quando a URL a nomeia", async () => {
    routeTo("/biblia/efesios/5");
    expect(await screen.findByRole("heading", { name: "Efésios 5" })).toBeInTheDocument();
    expect(await screen.findByText("13 sermões")).toBeInTheDocument();
    expect(screen.getByText(/O casamento/)).toBeInTheDocument();
  });

  it("pede ao servidor exatamente a passagem pedida", async () => {
    routeTo("/biblia/efesios/5");
    await screen.findByText("13 sermões");

    const calls = (globalThis.fetch as unknown as { mock: { calls: string[][] } }).mock.calls;
    const listing = calls.map((c) => c[0]).find((u) => u?.startsWith("/api/sermons"));
    expect(listing).toContain("livros=efesios");
    expect(listing).toContain("capitulo=5");
  });
});

describe("/series", () => {
  it("ordena os capítulos da Confissão numericamente", async () => {
    // "CFW 23" must not fall between "CFW 2" and "CFW 3".
    routeTo("/series");
    await screen.findByText(/CFW 2 —/);

    const names = screen
      .getAllByRole("link")
      .map((l) => l.textContent ?? "")
      .filter((t) => t.startsWith("CFW"));
    expect(names[0]).toContain("CFW 2");
    expect(names[1]).toContain("CFW 3");
    expect(names[2]).toContain("CFW 23");
  });

  it("separa as aulas avulsas das séries de verdade", async () => {
    routeTo("/series");
    expect(await screen.findByText("Avulsos")).toBeInTheDocument();
  });
});

describe("/pregadores", () => {
  it("agrupa pelo cargo e mostra só o nome", async () => {
    routeTo("/pregadores");
    expect(await screen.findByText("Reverendo")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Bruno Melo, 268 sermões/ })).toBeInTheDocument();
  });
});

describe("/datas", () => {
  it("lista os meses do ano mais recente primeiro", async () => {
    routeTo("/datas");
    expect(await screen.findByText("2024")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Março, 9 sermões/ })).toBeInTheDocument();
  });

  it("monta a janela de datas do mês escolhido", async () => {
    routeTo("/datas/2024/3");
    await screen.findByRole("heading", { name: /Março de 2024/ });

    const calls = (globalThis.fetch as unknown as { mock: { calls: string[][] } }).mock.calls;
    const listing = calls.map((c) => c[0]).find((u) => u?.startsWith("/api/sermons"));
    expect(listing).toContain("de=2024-03-01");
    expect(listing).toContain("ate=2024-03-31");
  });
});

describe("/temas", () => {
  it("agrupa os tópicos pelo seu grupo", async () => {
    routeTo("/temas");
    expect(await screen.findByText("Vida Cristã")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Ansiedade, 4 sermões/ })).toBeInTheDocument();
  });

  it("abre o tópico e mostra o grupo como subtítulo", async () => {
    routeTo("/temas/ansiedade");
    expect(await screen.findByRole("heading", { name: "Ansiedade" })).toBeInTheDocument();
  });

  it("explica que ainda não há temas em vez de fingir que o acervo é vazio", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url.startsWith("/api/facets")
          ? okResponse({ ...FACETS, temas: [] })
          : okResponse({ total: 0, sermons: [], pagina: 1 }),
      ),
    );
    routeTo("/temas");
    expect(await screen.findByText(/temas ainda estão sendo preparados/i)).toBeInTheDocument();
  });
});
